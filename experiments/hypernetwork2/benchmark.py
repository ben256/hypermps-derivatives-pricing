import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoreDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            r: int,
    ):
        super().__init__()
        self.r = r
        self.output_size = 2 * r * r  # 2 cores (for bit 0 and bit 1)

        self.fc = nn.Linear(hidden_size, self.output_size)
        self.activation = nn.Tanh()

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x: [batch_size, hidden_size]
        Returns: g0, g1 each of shape [batch_size, r, r]
        """
        x = self.fc(x)
        x = self.activation(x)

        # Split into g0 and g1
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
    ):
        """
        Decoder that outputs all L cores for a single dimension.

        Args:
            hidden_size: size of the hidden representation
            r: TT rank
            L: number of bit levels (cores per dimension)
        """
        super().__init__()
        self.r = r
        self.L = L
        # Output 2*L cores (L cores for bit=0, L cores for bit=1)
        self.output_size = 2 * L * r * r

        self.fc = nn.Linear(hidden_size, self.output_size)
        self.activation = nn.Tanh()

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x: [batch_size, hidden_size]
        Returns: tuple of (cores_0, cores_1)
            cores_0: list of L tensors, each [batch_size, r, r] for bit=0
            cores_1: list of L tensors, each [batch_size, r, r] for bit=1
        """
        batch_size = x.size(0)
        x = self.fc(x)
        x = self.activation(x)

        # Reshape to [batch_size, 2, L, r, r]
        x = x.view(batch_size, 2, self.L, self.r, self.r)

        # Split into cores for bit=0 and bit=1
        cores_0 = [x[:, 0, l, :, :] for l in range(self.L)]  # L tensors of [batch_size, r, r]
        cores_1 = [x[:, 1, l, :, :] for l in range(self.L)]  # L tensors of [batch_size, r, r]

        return cores_0, cores_1


class SharedDecoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            r: int,
            K: int,
    ):
        """
        Fully shared decoder that outputs all K cores from a single linear layer.
        This demonstrates the difficulty of outputting many parameters from one decoder.

        Args:
            hidden_size: size of the hidden representation
            r: TT rank
            K: total number of cores (d × L)
        """
        super().__init__()
        self.r = r
        self.K = K
        # Output 2*K cores (K cores for bit=0, K cores for bit=1)
        self.output_size = 2 * K * r * r

        self.fc = nn.Linear(hidden_size, self.output_size)
        self.activation = nn.Tanh()

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x: [batch_size, hidden_size]
        Returns: tuple of (cores_0, cores_1)
            cores_0: list of K tensors, each [batch_size, r, r] for bit=0
            cores_1: list of K tensors, each [batch_size, r, r] for bit=1
        """
        batch_size = x.size(0)
        x = self.fc(x)
        x = self.activation(x)

        # Reshape to [batch_size, 2, K, r, r]
        x = x.view(batch_size, 2, self.K, self.r, self.r)

        # Split into cores for bit=0 and bit=1
        cores_0 = [x[:, 0, k, :, :] for k in range(self.K)]  # K tensors of [batch_size, r, r]
        cores_1 = [x[:, 1, k, :, :] for k in range(self.K)]  # K tensors of [batch_size, r, r]

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
    ):
        """
        Simple FNN-based MPS model for benchmark comparison.

        Args:
            N: grid size
            r: max TT rank
            d: dimensionality
            conditioning_dim: dimension of conditioning parameters
            hidden_size: size of hidden layers
            num_layers: number of hidden layers
            dropout: dropout rate
            decoder_type: 'core' for one decoder per core (K decoders),
                         'dimension' for one decoder per dimension (d decoders),
                         'shared' for one decoder for all cores (1 decoder)
        """
        super().__init__()

        L = int(math.log2(N))
        self.K = d * L
        self.r = r
        self.d = d
        self.L = L
        self.decoder_type = decoder_type

        # Input layer
        self.fc1 = nn.Linear(conditioning_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)
        ])

        # Decoders - either per-core, per-dimension, or fully shared
        if decoder_type == 'core':
            # One decoder for each position in the TT chain (K total)
            self.core_decoders = nn.ModuleList([
                CoreDecoder(hidden_size, r) for _ in range(self.K)
            ])
            self.dimension_decoders = None
            self.shared_decoder = None
        elif decoder_type == 'dimension':
            # One decoder for each dimension (d total)
            self.dimension_decoders = nn.ModuleList([
                DimensionDecoder(hidden_size, r, L) for _ in range(d)
            ])
            self.core_decoders = None
            self.shared_decoder = None
        elif decoder_type == 'shared':
            # One decoder for all cores (1 total)
            self.shared_decoder = SharedDecoder(hidden_size, r, self.K)
            self.core_decoders = None
            self.dimension_decoders = None
        else:
            raise ValueError(f"decoder_type must be 'core', 'dimension', or 'shared', got {decoder_type}")

        # Boundary vectors
        scale = (r ** -0.5)
        self.u = nn.Parameter(torch.ones(r) * scale)
        self.v = nn.Parameter(torch.ones(r) * scale)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
            self,
            conditioning_params: torch.Tensor,
            bits: torch.Tensor,
    ):
        """
        Forward pass.

        Args:
            conditioning_params: [batch_size, conditioning_dim]
            bits: [batch_size, S, K] binary bits

        Returns:
            [batch_size, S] predicted log-densities
        """
        batch_size, S, _ = bits.shape

        # Process conditioning parameters through FNN
        x = self.fc1(conditioning_params)
        x = self.relu(x)
        x = self.dropout(x)

        for layer in self.hidden_layers:
            x = layer(x)

        # Generate all cores based on decoder type
        cores_0 = []  # cores for bit=0
        cores_1 = []  # cores for bit=1

        if self.decoder_type == 'core':
            # Generate cores using per-core decoders
            for k in range(self.K):
                g0, g1 = self.core_decoders[k](x)
                cores_0.append(g0)
                cores_1.append(g1)

        elif self.decoder_type == 'dimension':
            # Generate cores using per-dimension decoders
            for dim in range(self.d):
                dim_cores_0, dim_cores_1 = self.dimension_decoders[dim](x)
                cores_0.extend(dim_cores_0)  # Add all L cores for this dimension
                cores_1.extend(dim_cores_1)

        elif self.decoder_type == 'shared':
            # Generate all cores from a single shared decoder
            cores_0, cores_1 = self.shared_decoder(x)

        # Stack cores: [batch_size, K, r, r]
        g0_all = torch.stack(cores_0, dim=1)
        g1_all = torch.stack(cores_1, dim=1)

        # Add residual connection
        rs = F.softplus(self.res_scale)
        I = torch.eye(self.r, device=g0_all.device, dtype=g0_all.dtype).view(1, 1, self.r, self.r)
        g0_all = g0_all + rs * I
        g1_all = g1_all + rs * I

        # Tensor train contraction
        L = self.u.view(1, 1, 1, self.r).expand(batch_size, S, 1, self.r)

        for k in range(self.K):
            gk0 = g0_all[:, k, :, :]  # [batch_size, r, r]
            gk1 = g1_all[:, k, :, :]  # [batch_size, r, r]
            bk = bits[:, :, k]  # [batch_size, S]

            # Select core based on bit value (0 or 1)
            Gk_sel = torch.where(
                bk.unsqueeze(-1).unsqueeze(-1).bool(),
                gk1.unsqueeze(1),  # [batch_size, 1, r, r] -> [batch_size, S, r, r]
                gk0.unsqueeze(1),  # [batch_size, 1, r, r] -> [batch_size, S, r, r]
            )

            L = torch.matmul(L, Gk_sel)

        out = torch.matmul(L, self.v.view(1, 1, self.r, 1))
        return out.view(batch_size, S)  # [B, S]

    def orth_loss(self):
        """No orthogonality loss for this simple baseline"""
        return 0.0
