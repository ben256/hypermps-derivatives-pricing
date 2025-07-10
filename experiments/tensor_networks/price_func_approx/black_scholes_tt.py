import numpy as np
import torch
import tntorch as tn
from scipy.stats import norm


def bs_call(S, K, T, r, sigma):

    if T == 0:
        return max(0.0, S - K)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    return call_price


def norm_cdf(x):
    return 0.5 * (1 - torch.erf(-x / np.sqrt(2)))


def bs_call_array(S, K, T, r, sigma):

    # T = torch.where(T == 0, torch.full_like(T, 1e-12), T)

    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)

    price = S * norm.cdf(d1) - K * torch.exp(-r * T) * norm.cdf(d2)
    # price = S * norm_cdf(d1) - K * torch.exp(-r * T) * norm_cdf(d2)

    final_price = torch.where(T == 0, torch.clamp(S - K, min=0), price)  # T == 0 case

    return final_price


def price_from_tt(S_query, T_query, sigma_query, tt_tensor, param_deltas):

    i_S = int(round((S_query - S_min) / param_deltas[0]))
    i_T = int(round((T_query - T_min) / param_deltas[1]))
    i_sigma = int(round((sigma_query - sigma_min) / param_deltas[2]))

    num_points = base**basis_size
    i_S = max(0, min(i_S, num_points - 1))
    i_T = max(0, min(i_T, num_points - 1))
    i_sigma = max(0, min(i_sigma, num_points - 1))

    price_tt = tt_tensor[i_S, i_T, i_sigma]

    return price_tt.item()


def price_from_tt_2(S_query, T_query, sigma_query, tt_tensor, param_deltas):

    i_S = int(round((S_query - S_min) / param_deltas[0]))
    i_T = int(round((T_query - T_min) / param_deltas[1]))
    i_sigma = int(round((sigma_query - sigma_min) / param_deltas[2]))

    num_pts = base**basis_size
    i_S = max(0, min(i_S, num_pts - 1))
    i_T = max(0, min(i_T, num_pts - 1))
    i_sigma = max(0, min(i_sigma, num_pts - 1))

    grid_indices = [i_S, i_T, i_sigma]
    tt_indices = []

    for grid_index in grid_indices:
        for digit in str(grid_index)[::-1]:
            tt_indices.append(int(digit))

    price_tt = tt_tensor[tuple(tt_indices)].item()

    return price_tt


K = 100.0
r = 0.05

S_min, S_max = 80.0, 120.0
T_min, T_max = 0.0, 1.0
sigma_min, sigma_max = 0.1, 0.4
params = {
    'S': [S_min, S_max],
    'T': [T_min, T_max],
    'sigma': [sigma_min, sigma_max],
}

base = 10
basis_size = 3

N = len(params)
coefficients = [base ** i for i in range(basis_size)]
param_deltas = [(v[1] - v[0]) / (np.sum(coefficients) * (base - 1)) for v in params.values()]


def black_scholes_wrapper(*args):
    z = torch.stack(args)
    z = torch.reshape(z, (N, basis_size, -1))

    indices = torch.tensordot(z, torch.tensor(coefficients, dtype=z.dtype), dims=([1], [0]))
    S = params['S'][0] + indices[0] * param_deltas[0]
    T = params['T'][0] + indices[1] * param_deltas[1]
    sigma = params['sigma'][0] + indices[2] * param_deltas[2]

    price = bs_call_array(S, K, T, r, sigma)
    return price


domain = [torch.arange(0, base) for _ in range(N * basis_size)]

print("Starting TT-Cross")
t_bs = tn.cross(
    function=black_scholes_wrapper,
    domain=domain,
    eps=1e-6,
    rmax=15,
    max_iter=10
)
print("TT-ranks:", t_bs.ranks_tt)

S_test, T_test, sigma_test = 105.0, 0.5, 0.25
tt_price = price_from_tt_2(S_test, T_test, sigma_test, t_bs, param_deltas)
analytical_price = bs_call(S_test, K, T_test, r, sigma_test)

print(f"\nParams: S={S_test}, T={T_test}, sigma={sigma_test}")
print(f"TT Price: {tt_price:.6f}")
print(f"Analytical Price: {analytical_price:.6f}")
print(f"Error: {abs(tt_price - analytical_price):.2e}")
