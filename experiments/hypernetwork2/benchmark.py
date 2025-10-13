import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def orth_loss(self):
        return 0.0
