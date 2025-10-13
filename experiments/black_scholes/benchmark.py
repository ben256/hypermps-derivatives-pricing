import math

import torch
from torch import nn
import torch.nn.functional as F

from qtt_functions import QTT


def _make_decoder(in_dim: int, out_dim: int, depth: int) -> nn.Module:
    """Create MLP decoder with specified depth."""
    layers = []
    if depth <= 1:
        layers.append(nn.Linear(in_dim, out_dim))
    else:
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())
        for _ in range(depth - 2):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*layers)


class CoreDecoder(nn.Module):
    """Separate decoder for each individual core."""
    def __init__(self, hidden_size: int, r: int, depth: int):
        super().__init__()
        self.r = r
        self.net = _make_decoder(hidden_size, 2 * r * r, depth)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        cores = self.net(h)
        return cores.view(h.size(0), 2, self.r, self.r)


class DimensionDecoder(nn.Module):
    """Decoder for all cores in one dimension."""
    def __init__(self, hidden_size: int, r: int, L: int, depth: int):
        super().__init__()
        self.r = r
        self.L = L
        self.net = _make_decoder(hidden_size, 2 * L * r * r, depth)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        cores = self.net(h)
        return cores.view(h.size(0), self.L, 2, self.r, self.r)


class SharedDecoder(nn.Module):
    """Shared decoder for all QTT cores."""
    def __init__(self, hidden_size: int, r: int, K: int, depth: int):
        super().__init__()
        self.r = r
        self.K = K
        self.net = _make_decoder(hidden_size, 2 * K * r * r, depth)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        cores = self.net(h)
        return cores.view(h.size(0), self.K, 2, self.r, self.r)


class BenchmarkMPS(nn.Module):
    """Benchmark MPS model for Black-Scholes density approximation.
    
    Supports three decoder architectures:
    - 'core': separate decoder for each core (most parameters)
    - 'dimension': separate decoder per dimension (medium parameters)
    - 'shared': single shared decoder for all cores (fewest parameters)
    """
    def __init__(
            self,
            N: int,
            r: int,
            d: int,
            conditioning_dim: int,
            hidden_size: int = 256,
            num_layers: int = 2,
            num_decoder_layers: int = 1,
        decoder_type: str = 'shared',
        use_core_softmax: bool = False,
        use_bit_mixing: bool = False,
        bit_mix_init: float = 0.1,
    ):
        super().__init__()

        L = int(math.log2(N))
        self.N = N
        self.K = d * L
        self.r = r
        self.d = d
        self.L = L
        self.decoder_type = decoder_type
        self.eps = 1e-6
        self.use_core_softmax = use_core_softmax
        self.use_bit_mixing = use_bit_mixing
        bit_mix_init = float(max(0.0, min(0.5, bit_mix_init)))
        self.bit_mix_logits = nn.Parameter(
            torch.full((self.K,), math.log(bit_mix_init / max(1e-6, 0.5 - bit_mix_init)))
        ) if use_bit_mixing else None

        # Trunk network
        trunk_layers = [nn.Linear(conditioning_dim, hidden_size), nn.GELU()]
        for _ in range(max(0, num_layers - 1)):
            trunk_layers.append(nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(nn.GELU())
        self.trunk = nn.Sequential(*trunk_layers)

        if decoder_type == 'core':
            self.core_decoders = nn.ModuleList([
                CoreDecoder(hidden_size, r, num_decoder_layers) for _ in range(self.K)
            ])
        elif decoder_type == 'dimension':
            self.dimension_decoders = nn.ModuleList([
                DimensionDecoder(hidden_size, r, L, num_decoder_layers) for _ in range(d)
            ])
        elif decoder_type == 'shared':
            self.shared_decoder = SharedDecoder(hidden_size, r, self.K, num_decoder_layers)
        else:
            raise ValueError(f"decoder_type must be 'core', 'dimension', or 'shared', got {decoder_type}")

        # Boundary vectors
        self.left_boundary = nn.Parameter(torch.zeros(r))
        self.right_boundary = nn.Parameter(torch.zeros(r))

    def _decode_cores(self, h: torch.Tensor) -> torch.Tensor:
        """Decode trunk output to QTT cores based on decoder type."""
        if self.decoder_type == 'core':
            cores_list = [decoder(h) for decoder in self.core_decoders]
            cores = torch.stack(cores_list, dim=1)  # [B, K, 2, r, r]
        elif self.decoder_type == 'dimension':
            dim_cores = [decoder(h) for decoder in self.dimension_decoders]
            cores = torch.cat(dim_cores, dim=1)  # [B, K, 2, r, r]
        else:  # shared
            cores = self.shared_decoder(h)  # [B, K, 2, r, r]
        return cores

    @staticmethod
    def _softmax_stable(x: torch.Tensor, dim: int) -> torch.Tensor:
        x = x - x.amax(dim=dim, keepdim=True)
        return torch.softmax(x, dim=dim)

    def forward(self, conditioning_params: torch.Tensor):
        """Forward pass: conditioning -> QTT cores."""
        h = self.trunk(conditioning_params)
        cores = self._decode_cores(h)
        if self.use_core_softmax:
            cores = self._softmax_stable(cores, dim=2).clamp_min(self.eps)
            left = self._softmax_stable(self.left_boundary, dim=0).clamp_min(self.eps)
            right = self._softmax_stable(self.right_boundary, dim=0).clamp_min(self.eps)
        else:
            cores = F.softplus(cores) + self.eps
            left = F.softplus(self.left_boundary) + self.eps
            right = F.softplus(self.right_boundary) + self.eps

        # Optional bit mixing to reduce hard dyadic splits (mitigates spikes/notches at core boundaries like z≈0)
        if self.use_bit_mixing and (self.bit_mix_logits is not None):
            # m_k in (0, 0.5); mix matrix [[1-m, m],[m,1-m]]
            m = 0.5 * torch.sigmoid(self.bit_mix_logits)  # shape [K]
            mixed_cores = []
            for k in range(self.K):
                mk = m[k]
                mix = torch.stack([
                    torch.stack([1.0 - mk, mk]),
                    torch.stack([mk, 1.0 - mk])
                ], dim=0).to(cores.device).to(cores.dtype)  # [2,2]
                # apply along bit dimension
                # cores[:, k] has shape [B, 2, r, r]
                ck = cores[:, k]
                # new_bit[b] = sum_b2 mix[bit,b2]*ck[b2]
                new_ck = torch.einsum('ab,bxij->axij', mix, ck.permute(1,0,2,3))  # [2,B,r,r]
                new_ck = new_ck.permute(1,0,2,3).contiguous()  # [B,2,r,r]
                mixed_cores.append(new_ck)
            cores = torch.stack(mixed_cores, dim=1)  # [B,K,2,r,r]

        qtts = []
        for b in range(conditioning_params.size(0)):
            per_core = [cores[b, k] for k in range(self.K)]
            qtts.append(QTT(u=left, cores=per_core, v=right))

        if conditioning_params.size(0) == 1:
            return qtts[0]
        return qtts

    # def contract_qtt(self, conditioning_params: torch.Tensor) -> torch.Tensor:
    #     h = self.trunk(conditioning_params)
    #     cores = self._decode_cores(h)
    #     cores = self._softmax_stable(cores, dim=2).clamp_min(self.eps)
    #
    #     left = F.softplus(self.left_boundary) + self.eps
    #     right = F.softplus(self.right_boundary) + self.eps
    #
    #     batch_size = conditioning_params.size(0)
    #     num_entries = self.N if self.d == 1 else 1 << self.K
    #     contracted = conditioning_params.new_empty((batch_size, num_entries))
    #
    #     for b in range(batch_size):
    #         states = left.unsqueeze(0)  # [S, r], starts with S = 1
    #         for core in cores[b]:
    #             # Combine current states with both bit slices of the core
    #             # and flatten back to shape [2*S, r].
    #             next_states = torch.einsum('sr,brc->sbc', states, core)
    #             states = next_states.reshape(-1, self.r)
    #         contracted[b] = states @ right
    #
    #     return contracted
