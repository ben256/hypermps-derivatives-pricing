import math
import numpy as np
import torch
import tntorch as tn
from scipy.stats import norm

from utils import bs_density_conditioning_param_len


def sample_bs_params_batch(batch_size: int, rng: torch.Generator):
    """Sample Black-Scholes parameters with precomputed m and s."""
    conditioning_dim = bs_density_conditioning_param_len(1)
    conditioning_params = torch.zeros(batch_size, conditioning_dim, dtype=torch.float32)
    params_list = []

    for i in range(batch_size):
        S0 = torch.empty(1).uniform_(80.0, 120.0, generator=rng).item()
        K = torch.empty(1).uniform_(80.0, 120.0, generator=rng).item()
        r = torch.empty(1).uniform_(0.01, 0.1, generator=rng).item()
        sigma = torch.empty(1).uniform_(0.1, 0.5, generator=rng).item()
        T = torch.empty(1).uniform_(0.25, 2.0, generator=rng).item()

        # Precompute log-space parameters
        m = math.log(S0) + (r - 0.5 * sigma * sigma) * T
        s = sigma * math.sqrt(T)

        conditioning_params[i] = torch.tensor([S0, r, sigma, T], dtype=torch.float32)
        params_list.append({
            "S0": S0,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "m": m,
            "s": s,
        })

    return conditioning_params, params_list


def bs_call_price_analytical(S0: float, K: float, r: float, sigma: float, T: float):
    """Analytical Black-Scholes call price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price_analytical(S0: float, K: float, r: float, sigma: float, T: float):
    """Analytical Black-Scholes put price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def qtt_fold(vec: torch.Tensor, L: int) -> torch.Tensor:
    """Fold a vector into QTT shape."""
    q = vec.view(*([2]*L))
    # bit-reversal permutation: [L-1, L-2, ..., 0]
    return q.permute(*reversed(range(L)))


def build_payoff_qtt(z: torch.Tensor, dz: torch.Tensor, K: float, m: torch.Tensor, s: torch.Tensor, L: int) -> tn.Tensor:
    """Build call payoff tensor in standardized log-price space."""
    S = torch.exp(m + s * z)
    payoff = torch.clamp(S - K, min=0.0) * dz
    return tn.Tensor(qtt_fold(payoff, L))


def build_put_payoff_qtt(z: torch.Tensor, dz: torch.Tensor, K: float, m: torch.Tensor, s: torch.Tensor, L: int) -> tn.Tensor:
    """Build put payoff tensor in standardized log-price space."""
    S = torch.exp(m + s * z)
    payoff = torch.clamp(K - S, min=0.0) * dz
    return tn.Tensor(qtt_fold(payoff, L))


def standard_normal_pdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    inv_sqrt_2pi = z.new_tensor(1.0 / math.sqrt(2 * math.pi))
    return inv_sqrt_2pi * torch.exp(-0.5 * z ** 2)
