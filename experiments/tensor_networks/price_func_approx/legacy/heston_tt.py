from datetime import datetime
import argparse

import numpy as np
import torch
import tntorch as tn

from utils.heston_fft import heston_pricer_fft


def price_from_tt2(S, K, T, v0, tt_tensor, grid_info):

    variable_params = grid_info['variable_params']
    param_deltas = grid_info['param_deltas']
    base = grid_info['base']
    basis_size = grid_info['basis_size']

    params = np.stack([S, K, T, v0], axis=0)
    param_mins = np.array([v[0] for v in variable_params.values()]).reshape(-1, 1)
    deltas = np.array(param_deltas).reshape(-1, 1)

    grid_indices = np.round((params - param_mins) / deltas).astype(np.int64)

    num_points = base ** basis_size
    grid_indices = np.clip(grid_indices, 0, num_points - 1)

    powers = base ** np.arange(basis_size - 1, -1, -1, dtype=np.int64)

    tt_indices_per_param = (grid_indices.T[:, :, np.newaxis] // powers) % base

    num_samples = S.shape[0]
    num_tt_dims = len(variable_params) * basis_size
    tt_indices = tt_indices_per_param.reshape(num_samples, num_tt_dims)

    prices = tt_tensor[tt_indices].numpy()

    return prices


def approximate_heston_price_with_tt(
        tt_tolerance: float,
        tt_rmax: int,
        tt_max_iter: int,
        test_values: tuple = None,
        test_whole_grid: bool = False,
        base: int = 10,
        basis_size: int = 2,
        save_folder: str = './output/tensor_trains',
        S_limit: tuple = (90.0, 110.0),
        K_limit: tuple = (90.0, 110.0),
        T_limit: tuple = (0.01, 1.0),
        v0_limit: tuple = (0.01, 0.04),
        kappa: float = 1.0,
        theta: float = 0.02,
        rho: float = -0.7,
        sigma_v: float = 0.1,
        r: float = 0.05,
        verbose: bool = False
):

    # Variable Parameters
    S_min, S_max = S_limit
    K_min, K_max = K_limit
    T_min, T_max = T_limit
    v0_min, v0_max = v0_limit
    variable_params = {
        'S': [S_min, S_max],
        'K': [K_min, K_max],
        'T': [T_min, T_max],
        'v0': [v0_min, v0_max],
    }

    # Domain Setup
    N = len(variable_params)
    domain = [torch.arange(0, base) for _ in range(N * basis_size)]
    coefficients = [base ** i for i in range(basis_size)]
    param_deltas = [(v[1] - v[0]) / (np.sum(coefficients) * (base - 1)) for v in variable_params.values()]


    def heston_wrapper(*args):
        z = torch.stack(args)
        z = torch.reshape(z, (N, basis_size, -1))

        indices = torch.tensordot(z, torch.tensor(coefficients, dtype=z.dtype), dims=([1], [0]))

        S_np = (S_min + indices[0] * param_deltas[0]).numpy()
        K_np = (K_min + indices[1] * param_deltas[1]).numpy()
        T_np = (T_min + indices[2] * param_deltas[2]).numpy()
        v0_np = (v0_min + indices[3] * param_deltas[3]).numpy()

        prices_np = heston_pricer_fft(S_np, K_np, T_np, sigma_v, kappa, rho, theta, v0_np, r, 0.0)

        return torch.from_numpy(prices_np)


    tt_heston, tt_cross_info = tn.cross(
        function=heston_wrapper,
        domain=domain,
        eps=tt_tolerance,
        rmax=tt_rmax,
        max_iter=tt_max_iter,
        early_stopping_patience=3,
        early_stopping_tolerance=1e-7,
        return_info=True,
    )

    grid_info = {
        'variable_params': variable_params,
        'param_deltas': param_deltas,
        'base': base,
        'basis_size': basis_size
    }

    dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
    torch.save({'tensor': tt_heston, 'grid_info': grid_info}, f'{save_folder}/heston_tt_tensor_{dt_str}.pt')

    if test_whole_grid:
        grid_ranges = [np.linspace(lim[0], lim[1], 10) for lim in (S_limit, K_limit, T_limit, v0_limit)]
        grid_coords = np.stack(np.meshgrid(*grid_ranges, indexing='ij'), axis=-1)
        S_test, K_test, T_test, v0_test = grid_coords.reshape(-1, 4).T

        tt_prices = price_from_tt2(S_test, K_test, T_test, v0_test, tt_heston, grid_info)

    else:
        if test_values is None:
            raise("Test values must be provided when test_whole_grid is False.")

        S_test, K_test, T_test, v0_test = test_values
        tt_prices = price_from_tt2(S_test, K_test, T_test, v0_test, tt_heston, grid_info)

    return tt_heston, tt_cross_info, tt_prices



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tt_tolerance', type=float, default=1e-4)
    parser.add_argument('--tt_rmax', type=int, default=10)
    parser.add_argument('--tt_max_iter', type=int, default=2)

    args = parser.parse_args()
    approximate_heston_price_with_tt(tt_tolerance=args.tt_tolerance, tt_rmax=args.tt_rmax, tt_max_iter=args.tt_max_iter,
                                     test_values=([], [], [], []))
