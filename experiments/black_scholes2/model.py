import math
import torch
from torch import nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            n_heads: int,
            dropout: float,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, max(512, 4 * hidden_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(512, 4 * hidden_dim), hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            conditioning_tokens: torch.Tensor,
    ):
        query = self.ln1(x)
        attn_out, _ = self.cross_attn(
            query=query,
            key=conditioning_tokens,
            value=conditioning_tokens,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)

        y = self.ln2(x)
        y = self.ff(y)
        x = x + self.drop2(y)
        return x


class SequenceModel(nn.Module):
    def __init__(
            self,
            d: int,
            L: int,
            conditioning_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            M: int,
            n_layers: int,
            n_heads: int,
            dropout: float,
            conditioning_tokens: int,
    ):
        super().__init__()
        self.M = M
        self.d = d
        self.L = L
        self.hidden_dim = hidden_dim
        self.conditioning_tokens = conditioning_tokens

        self.dimension_emb = nn.Embedding(d, embedding_dim)
        self.bit_level_emb = nn.Embedding(L, embedding_dim)
        self.position_proj = nn.Linear(embedding_dim, hidden_dim)

        tokens = [(j, l) for j in range(d) for l in range(L)]
        self.register_buffer('dimension_idx', torch.tensor([t[0] for t in tokens], dtype=torch.long))
        self.register_buffer('bit_level_idx', torch.tensor([t[1] for t in tokens], dtype=torch.long))

        self.conditioning_proj = nn.Linear(conditioning_dim, self.conditioning_tokens * hidden_dim)

        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, M),
        )

    def forward(
            self,
            conditioning_params: torch.Tensor,
            batch_size: int,
    ):
        x = self.dimension_emb(self.dimension_idx) + self.bit_level_emb(self.bit_level_idx)  # [K, embedding_dim]
        x = self.position_proj(x).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, K, hidden_dim]

        conditioning_tokens = self.conditioning_proj(conditioning_params).view(
            batch_size, self.conditioning_tokens, self.hidden_dim
        )  # [B, conditioning_tokens, hidden_dim]

        h = x
        for block in self.blocks:
            h = block(h, conditioning_tokens)

        h = self.final_ln(h)

        logits = self.head(h)  # [batch_size, K, M]
        alpha = torch.softmax(logits, dim=-1)  # [batch_size, K, M]
        return alpha


class CoreBank(nn.Module):
    def __init__(
            self,
            r: int = 20,
            M: int = 8,
            orth_penalty: float = 5e-5,
    ):
        """
        r: max TT rank
        M: number of basis cores
        """
        super().__init__()
        self.B = nn.Parameter(torch.randn(M, 2, r, r) / (r ** 0.5))  # [M, 2, r, r]
        self.orth_penalty = orth_penalty

    def make_core(
            self,
            alpha: torch.Tensor
    ):
        """
        alpha: mixing weights for each core, shape [batch_size * k, M].
        """
        g0 = torch.einsum('bm,mij->bij', alpha, self.B[:, 0, :, :])
        g1 = torch.einsum('bm,mij->bij', alpha, self.B[:, 1, :, :])
        return g0, g1

    def orth_loss(
            self
    ):
        """
        Orthogonality loss to encourage each basis core to be orthogonal.
        """
        loss = 0.0
        for bit in (0, 1):
            B_slice = self.B[:, bit, :, :]  # [M, r, r]
            BtB = torch.matmul(B_slice.transpose(1, -2), B_slice)  # [M, r, r]
            I = torch.eye(BtB.size(-1), device=BtB.device).expand_as(BtB)  # [M, r, r]
            loss += (BtB - I).pow(2).mean()
        return self.orth_penalty * loss


class HyperHyperNetwork(nn.Module):
    def __init__(
            self,
            N: int,
            r: int,
            M: int,
            orth_penalty: float,
            d: int,
            conditioning_dim: int,
            embedding_dim: int,
            hidden_dim: int,
            n_layers: int,
            n_heads: int,
            dropout: float,
            conditioning_tokens: int,
    ):
        """
        HyperHyperNetwork for Black-Scholes QTT approximation.

        N: grid size
        r: max TT rank
        M: number of basis cores
        d: dimension (1 for 1D Black-Scholes)
        """
        super().__init__()

        L = int(math.log2(N))

        self.K = d * L
        self.M = M
        self.r = r

        self.core_bank = CoreBank(
            r=r,
            M=M,
            orth_penalty=orth_penalty,
        )

        self.sequence_model = SequenceModel(
            d=d,
            L=L,
            conditioning_dim=conditioning_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            M=M,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            conditioning_tokens=conditioning_tokens,
        )

        scale = (r ** -0.5)
        self.u = nn.Parameter(torch.ones(r) * scale)
        self.v = nn.Parameter(torch.ones(r) * scale)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
            self,
            conditioning_params: torch.Tensor,
            bits: torch.Tensor,
    ):
        batch_size, S, _ = bits.shape
        alpha = self.sequence_model(conditioning_params, batch_size)

        # reshape to easily pass through einsum
        ak = alpha.reshape(batch_size * self.K, self.M)  # [batch_size, K, M]
        g0, g1 = self.core_bank.make_core(ak)  # each [batch_size * K, r, r]

        # reshape back
        g0 = g0.view(batch_size, self.K, self.r, self.r)  # [batch_size, K, r, r]
        g1 = g1.view(batch_size, self.K, self.r, self.r)  # [batch_size, K, r, r]

        # add residual connection
        rs = F.softplus(self.res_scale)
        I = torch.eye(self.r, device=g0.device, dtype=g0.dtype).view(1, 1, self.r, self.r)
        g0 = g0 + rs * I
        g1 = g1 + rs * I

        # contraction!!
        L = self.u.view(1, 1, 1, self.r).expand(batch_size, S, 1, self.r)
        for k in range(self.K):
            gk0 = g0[:, k, :, :]
            gk1 = g1[:, k, :, :]
            bk = bits[:, :, k]

            # select core based on bit value (0 or 1)
            Gk_sel = torch.where(
                bk.unsqueeze(-1).unsqueeze(-1).bool(),
                gk1.unsqueeze(1),  # [batch_size, 1, r, r] -> [batch_size, S, r, r]
                gk0.unsqueeze(1),  # [batch_size, 1, r, r] -> [batch_size, S, r, r]
            )

            L = torch.matmul(L, Gk_sel)

        out = torch.matmul(L, self.v.view(1, 1, self.r, 1))

        return out.view(batch_size, S)  # [B, S]

    def get_qtt(self, conditioning_params: torch.Tensor):
        """
        Returns the uncontracted QTT representation for the given batch of conditioning params.
        Shapes:
          - u: [B, r]
          - cores: [B, K, 2, r, r]  (bit axis: 0 then 1)
          - v: [B, r]
        """
        batch_size = conditioning_params.size(0)
        alpha = self.sequence_model(conditioning_params, batch_size)             # [B, K, M]

        # Build cores from bank
        ak = alpha.reshape(batch_size * self.K, self.M)                          # [B*K, M]
        g0, g1 = self.core_bank.make_core(ak)                                    # each [B*K, r, r]
        g0 = g0.view(batch_size, self.K, self.r, self.r)                         # [B, K, r, r]
        g1 = g1.view(batch_size, self.K, self.r, self.r)                         # [B, K, r, r]

        # Residual identity (matches forward)
        rs = F.softplus(self.res_scale)
        I = torch.eye(self.r, device=g0.device, dtype=g0.dtype).view(1, 1, self.r, self.r)
        g0 = g0 + rs * I
        g1 = g1 + rs * I

        # Stack bit dimension -> [B, K, 2, r, r]
        cores = torch.stack([g0, g1], dim=2)

        # Boundary vectors -> [B, r]
        u = self.u.view(1, self.r).expand(batch_size, -1).contiguous()
        v = self.v.view(1, self.r).expand(batch_size, -1).contiguous()
        return u, cores, v

    def orth_loss(self):
        return self.core_bank.orth_loss()


class CoreDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            r: int,
            num_decoder_layers: int = 1,
    ):
        super().__init__()
        self.r = r
        self.output_size = 2 * r * r

        layers = []
        in_size = hidden_size
        for _ in range(num_decoder_layers - 1):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        self.fc = nn.Linear(in_size, self.output_size)
        self.activation = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.hidden(x)
        x = self.fc(x)
        x = self.activation(x)

        cores = x.view(x.size(0), 2, self.r, self.r)
        g0 = cores[:, 0, :, :]
        g1 = cores[:, 1, :, :]
        return g0, g1


class DimensionDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            r: int,
            L: int,
            num_decoder_layers: int = 1,
    ):
        super().__init__()
        self.r = r
        self.L = L
        self.output_size = 2 * L * r * r

        layers = []
        in_size = hidden_size
        for _ in range(num_decoder_layers - 1):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        self.hidden = nn.Sequential(*layers) if layers else nn.Identity()
        self.fc = nn.Linear(in_size, self.output_size)
        self.activation = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.hidden(x)
        x = self.fc(x)
        x = self.activation(x)

        x = x.view(batch_size, 2, self.L, self.r, self.r)

        cores_0 = [x[:, 0, l, :, :] for l in range(self.L)]
        cores_1 = [x[:, 1, l, :, :] for l in range(self.L)]

        return cores_0, cores_1


class SharedDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            r: int,
            K: int,
    ):
        super().__init__()
        self.r = r
        self.K = K
        self.output_size = 2 * K * r * r

        self.fc = nn.Linear(hidden_size, self.output_size)
        self.activation = nn.Tanh()

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc(x)
        x = self.activation(x)

        x = x.view(batch_size, 2, self.K, self.r, self.r)

        cores_0 = [x[:, 0, k, :, :] for k in range(self.K)]
        cores_1 = [x[:, 1, k, :, :] for k in range(self.K)]

        return cores_0, cores_1


class BenchmarkMPS(nn.Module):
    def __init__(
            self,
            N: int,
            r: int,
            d: int,
            conditioning_dim: int,
            hidden_size: int = 256,
            num_layers: int = 3,
            dropout: float = 0.1,
            decoder_type: str = 'core',  # 'core', 'dimension', or 'shared'
            num_decoder_layers: int = None,
    ):
        super().__init__()

        L = int(math.log2(N))
        self.K = d * L
        self.r = r
        self.d = d
        self.L = L
        self.decoder_type = decoder_type

        if num_decoder_layers is None:
            num_decoder_layers = 1

        self.fc1 = nn.Linear(conditioning_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)
        ])

        if decoder_type == 'core':
            self.core_decoders = nn.ModuleList([
                CoreDecoder(hidden_size, r, num_decoder_layers) for _ in range(self.K)
            ])
            self.dimension_decoders = None
            self.shared_decoder = None
        elif decoder_type == 'dimension':
            self.dimension_decoders = nn.ModuleList([
                DimensionDecoder(hidden_size, r, L, num_decoder_layers) for _ in range(d)
            ])
            self.core_decoders = None
            self.shared_decoder = None
        elif decoder_type == 'shared':
            self.shared_decoder = SharedDecoder(hidden_size, r, self.K)
            self.core_decoders = None
            self.dimension_decoders = None
        else:
            raise ValueError(f"decoder_type must be 'core', 'dimension', or 'shared', got {decoder_type}")

        scale = (r ** -0.5)
        self.u = nn.Parameter(torch.ones(r) * scale)
        self.v = nn.Parameter(torch.ones(r) * scale)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
            self,
            conditioning_params: torch.Tensor,
            bits: torch.Tensor,
    ):
        batch_size, S, _ = bits.shape

        x = self.fc1(conditioning_params)
        x = self.relu(x)
        x = self.dropout(x)

        for layer in self.hidden_layers:
            x = layer(x)

        cores_0 = []
        cores_1 = []

        if self.decoder_type == 'core':
            for k in range(self.K):
                g0, g1 = self.core_decoders[k](x)
                cores_0.append(g0)
                cores_1.append(g1)

        elif self.decoder_type == 'dimension':
            for dim in range(self.d):
                dim_cores_0, dim_cores_1 = self.dimension_decoders[dim](x)
                cores_0.extend(dim_cores_0)
                cores_1.extend(dim_cores_1)

        elif self.decoder_type == 'shared':
            cores_0, cores_1 = self.shared_decoder(x)

        g0_all = torch.stack(cores_0, dim=1)
        g1_all = torch.stack(cores_1, dim=1)

        rs = F.softplus(self.res_scale)
        I = torch.eye(self.r, device=g0_all.device, dtype=g0_all.dtype).view(1, 1, self.r, self.r)
        g0_all = g0_all + rs * I
        g1_all = g1_all + rs * I

        L = self.u.view(1, 1, 1, self.r).expand(batch_size, S, 1, self.r)

        for k in range(self.K):
            gk0 = g0_all[:, k, :, :]
            gk1 = g1_all[:, k, :, :]
            bk = bits[:, :, k]

            Gk_sel = torch.where(
                bk.unsqueeze(-1).unsqueeze(-1).bool(),
                gk1.unsqueeze(1),
                gk0.unsqueeze(1),
            )

            L = torch.matmul(L, Gk_sel)

        out = torch.matmul(L, self.v.view(1, 1, self.r, 1))
        return out.view(batch_size, S)

    def get_qtt(self, conditioning_params: torch.Tensor):
        """
        Returns the uncontracted QTT representation for the given batch of conditioning params.
        Shapes:
          - u: [B, r]
          - cores: [B, K, 2, r, r]  (bit axis: 0 then 1)
          - v: [B, r]
        """
        batch_size = conditioning_params.size(0)

        # Same trunk as forward
        x = self.fc1(conditioning_params)
        x = self.relu(x)
        x = self.dropout(x)
        for layer in self.hidden_layers:
            x = layer(x)

        # Per-dimension decoders -> lists of length L with [B, r, r] tensors
        cores_0, cores_1 = [], []
        for dim in range(self.d):
            dim_cores_0, dim_cores_1 = self.dimension_decoders[dim](x)
            cores_0.extend(dim_cores_0)
            cores_1.extend(dim_cores_1)

        # Stack to [B, K, r, r]
        g0_all = torch.stack(cores_0, dim=1)
        g1_all = torch.stack(cores_1, dim=1)

        # Residual identity (matches forward)
        rs = F.softplus(self.res_scale)
        I = torch.eye(self.r, device=g0_all.device, dtype=g0_all.dtype).view(1, 1, self.r, self.r)
        g0_all = g0_all + rs * I
        g1_all = g1_all + rs * I

        # Stack bit dimension -> [B, K, 2, r, r]
        cores = torch.stack([g0_all, g1_all], dim=2)

        # Boundary vectors -> [B, r]
        u = self.u.view(1, self.r).expand(batch_size, -1).contiguous()
        v = self.v.view(1, self.r).expand(batch_size, -1).contiguous()
        return u, cores, v

    def orth_loss(self):
        return 0.0
