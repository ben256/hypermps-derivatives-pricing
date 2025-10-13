import numpy as np
import torch
from scipy.stats import norm

from utils import bs_density_conditioning_param_len


STRIKE_MIN = 80.0
STRIKE_MAX = 120.0


def sample_bs_params(
        rng: torch.Generator,
):
    S0 = torch.empty(1).uniform_(80.0, 120.0, generator=rng).item()
    K = torch.empty(1).uniform_(STRIKE_MIN, STRIKE_MAX, generator=rng).item()
    r = torch.empty(1).uniform_(0.01, 0.1, generator=rng).item()
    sigma = torch.empty(1).uniform_(0.1, 0.5, generator=rng).item()
    T = torch.empty(1).uniform_(0.25, 2.0, generator=rng).item()

    conditioning_params = torch.tensor([S0, r, sigma, T], dtype=torch.float32)

    param_dict = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
    }

    return conditioning_params, param_dict


def sample_bs_params_batch(
        batch_size: int,
        rng: torch.Generator,
):
    conditioning_dim = bs_density_conditioning_param_len(1)
    conditioning_params = torch.zeros(batch_size, conditioning_dim)
    params_list = []

    for i in range(batch_size):
        cond_params, param_dict = sample_bs_params(rng)
        conditioning_params[i] = cond_params
        params_list.append(param_dict)

    return conditioning_params, params_list


def bs_call_price_analytical(
        S0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def bs_call_payoff_qtt(
        S_T: torch.Tensor,
        K: float,
):
    return torch.maximum(S_T - K, torch.zeros_like(S_T))


def bs_terminal_density(
        S_T: torch.Tensor,
        S0: float,
        r: float,
        sigma: float,
        T: float,
):
    mu = torch.log(torch.tensor(S0)) + (r - 0.5 * sigma**2) * T
    var = sigma**2 * T

    log_density = -0.5 * torch.log(torch.tensor(2 * torch.pi * var)) - torch.log(S_T) - 0.5 * (torch.log(S_T) - mu)**2 / var
    return log_density


def price_from_log_density(log_density, indices, grid, K, r, T):
    # normalize log-pdf: ∫ p(S)dS = 1
    dx = grid[1] - grid[0]
    logZ = torch.logsumexp(log_density + torch.log(dx), dim=1, keepdim=True)
    log_pdf = log_density - logZ
    pdf = torch.exp(log_pdf)

    payoff = torch.clamp(grid - K, min=0.0).to(pdf.dtype)[None, :]  # [1,N]
    # (indices/sort are unnecessary: grid is already sorted)
    price = torch.trapz(pdf * payoff, grid, dim=1)
    return price * torch.exp(torch.tensor(-r*T, device=price.device, dtype=price.dtype))
