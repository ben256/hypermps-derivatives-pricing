from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, cast

import torch
import tntorch as tn

Tensor = torch.Tensor


@dataclass
class QTT:
    """Quantized Tensor-Train representation."""
    u: Tensor                 # [r]
    cores: List[Tensor]       # K items of shape [2, r, r]
    v: Tensor                 # [r]

    @property
    def rank(self) -> int:
        return int(self.u.numel())

    @property
    def K(self) -> int:
        return len(self.cores)

    def to(self, *args, **kwargs) -> "QTT":
        return QTT(
            u=self.u.to(*args, **kwargs),
            cores=[c.to(*args, **kwargs) for c in self.cores],
            v=self.v.to(*args, **kwargs),
        )


def tt_scale(x: QTT, alpha: Tensor | float) -> QTT:
    """Scale TT by a scalar by absorbing into the left boundary vector."""
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, dtype=x.u.dtype, device=x.u.device)
    return QTT(u=x.u * alpha, cores=x.cores, v=x.v)


def _qtt_to_tn(qtt: QTT) -> tn.Tensor:
    """Convert the custom QTT structure to a tntorch.Tensor."""
    K = qtt.K
    device = qtt.u.device
    dtype = qtt.u.dtype

    if K == 0:
        raise ValueError("QTT must contain at least one core")

    tt_cores: List[Tensor] = []

    if K == 1:
        core = torch.zeros(1, 2, 1, device=device, dtype=dtype)
        for bit in range(2):
            val = qtt.u @ qtt.cores[0][bit] @ qtt.v
            core[0, bit, 0] = val
        tt_cores.append(core)
        return tn.Tensor(tt_cores)

    # First core: [1, 2, r]
    first = torch.zeros(1, 2, qtt.rank, device=device, dtype=dtype)
    for bit in range(2):
        first[0, bit, :] = qtt.u @ qtt.cores[0][bit]
    tt_cores.append(first)

    # Middle cores: [r, 2, r]
    for k in range(1, K - 1):
        core = qtt.cores[k].permute(1, 0, 2).contiguous()
        tt_cores.append(core)

    # Last core: [r, 2, 1]
    last = torch.zeros(qtt.rank, 2, 1, device=device, dtype=dtype)
    for bit in range(2):
        last[:, bit, 0] = qtt.cores[-1][bit] @ qtt.v
    tt_cores.append(last)

    return tn.Tensor(tt_cores)


def normalize_density_qtt(qtt_density: QTT, volume_element: float) -> QTT:
    """Normalize QTT density to integrate to 1."""
    mass = cast(torch.Tensor, _qtt_to_tn(qtt_density).sum())
    scale = 1.0 / (mass.item() * volume_element + 1e-12)
    return tt_scale(qtt_density, qtt_density.u.new_tensor(scale))


def price_call_tt(qtt_density: QTT, payoff_tt: tn.Tensor, r: float, T: float) -> Tensor:
    """Compute call price from QTT density and payoff."""
    inner = cast(torch.Tensor, tn.dot(_qtt_to_tn(qtt_density), payoff_tt))
    return inner * math.exp(-r * T)


def make_z_grid(z_min: float, z_max: float, L: int, device=None, dtype=torch.float32):
    """Create uniform grid in standardized log-price space."""
    n_points = 2 ** L
    z = torch.linspace(z_min, z_max, n_points, device=device, dtype=dtype)
    dz = z[1] - z[0]
    return z, dz


def qtt_to_dense_vector(qtt: QTT) -> torch.Tensor:
    """Return the flattened tensor in natural (non bit-reversed) order."""
    tn_tensor = _qtt_to_tn(qtt).torch()
    if qtt.K == 1:
        return tn_tensor.view(-1)

    dims = tuple(reversed(range(qtt.K)))
    return tn_tensor.permute(dims).reshape(-1)