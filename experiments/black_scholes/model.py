import math
import torch
from torch import nn
import torch.nn.functional as F

from qtt_functions import QTT


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

    def forward(self, conditioning_params: torch.Tensor):
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

    def contraction(
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

    def orth_loss(self):
        return self.core_bank.orth_loss()
