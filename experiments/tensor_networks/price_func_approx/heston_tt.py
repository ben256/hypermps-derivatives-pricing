import numpy as np
import torch
import tntorch as tn

from utils.heston_fft import heston_pricer_fft


def build_heston_tt(
        base: int,
        basis_size: int,
        tt_tolerance: float,
        tt_rmax: int,
        tt_max_iter: int,
        S: float,
        v0: float,
        kappa: float,
        theta: float,
        rho: float,
        sigma_v: float,
        rate: float,
        div: float,
        k_limit: tuple = (-0.5, 0.5),
        T_limit: tuple = (0.01, 2.0),
        **kwargs
):
    """ Log-moneyness maturity evaluation grid """
    k_min, k_max = k_limit
    T_min, T_max = T_limit
    variable_params = {
        'k': [k_min, k_max],
        'T': [T_min, T_max],
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Domain Setup
    N = len(variable_params)
    domain = [torch.arange(0, base, device=device) for _ in range(N * basis_size)]
    coefficients = torch.tensor([base ** i for i in range(basis_size)], dtype=torch.float32, device=device)
    param_deltas = [((v[1] - v[0]) / (torch.sum(coefficients) * (base - 1))).item() for v in variable_params.values()]

    def heston_wrapper(*args):
        z = torch.stack(args)
        z = torch.reshape(z, (N, basis_size, -1))

        indices = torch.tensordot(z, coefficients, dims=([1], [0]))

        k_val = k_min + indices[0] * param_deltas[0]
        T_val = T_min + indices[1] * param_deltas[1]

        K_val = S * torch.exp(k_val)

        S_np = np.full(K_val.shape, S)
        K_np = K_val.cpu().numpy()
        T_np = T_val.cpu().numpy()

        prices_np = heston_pricer_fft(
            S_np, K_np, T_np, sigma_v, kappa, rho, theta, v0, rate, div
        )
        return torch.from_numpy(prices_np).to(device)

    tt_heston, tt_cross_info = tn.cross(
        function=heston_wrapper,
        domain=domain,
        device=device,
        eps=tt_tolerance,
        rmax=tt_rmax,
        max_iter=tt_max_iter,
        early_stopping_patience=2,
        early_stopping_tolerance=1e-7,
        return_info=True,
    )

    grid_info = {
        'variable_params': variable_params,
        'param_deltas': param_deltas,
        'base': base,
        'basis_size': basis_size,
        'S': S
    }

    return tt_heston, grid_info


def price_from_tt(K, T, tt_tensor, grid_info):
    variable_params = grid_info['variable_params']
    param_deltas = grid_info['param_deltas']
    base = grid_info['base']
    basis_size = grid_info['basis_size']
    S = grid_info['S']

    k = np.log(K / S)
    params = np.array([k, T])

    param_mins = np.array([v[0] for v in variable_params.values()])
    deltas = np.array(param_deltas)

    grid_indices = np.round((params - param_mins) / deltas).astype(np.int64)

    num_points = base ** basis_size
    grid_indices = np.clip(grid_indices, 0, num_points - 1)

    powers = base ** np.arange(0, basis_size, 1, dtype=np.int64)

    tt_indices_per_param = (grid_indices[:, np.newaxis] // powers) % base

    tt_indices = tt_indices_per_param.flatten().reshape(1, -1)

    price = tt_tensor[tt_indices].numpy()

    return price
