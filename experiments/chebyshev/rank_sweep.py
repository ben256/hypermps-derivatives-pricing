import argparse
import logging
import time
import random
import json

import numpy as np

from heston_chebyshev_tt import HestonChebyshevTTPricer


def perform_rank_sweep():
    """
    Double loop: iterate through the number of variable parameters (dimensions), then inside run TT-cross with different
    max ranks according to `ranks = np.arange(2, 35, 5)`. Saves results to JSON which is used in notebook to produce
    heatmap.
    """
    parameter_limits = {
        'S': [80.0, 120.0],
        'K': [80.0, 120.0],
        'T': [0.1, 2.0],
        'sigma_v': [0.1, 1.0],
        'kappa': [0.5, 5.0],
        'rho': [-0.9, -0.1],
        'theta': [0.01, 0.1],
        'v0': [0.01, 0.2],
        'rate': [0.01, 0.1],
        'div': [0.0, 0.05]
    }

    n_test = 100
    test_samples = {
        k: [random.uniform(v[0], v[1]) for _ in range(n_test)] for k, v in parameter_limits.items()
    }
    test_samples['batch_size'] = n_test

    ranks = np.arange(2, 35, 5)
    interpolation_order = 8
    n = len(parameter_limits.keys())
    dimensions = np.arange(1, n + 1)

    offline_times = []
    online_times = []
    analytical_times = []
    max_ranks = []
    max_abs_errors = []
    rms_errors = []
    n_samples = []

    for i in range(n):

        logging.info(f'Starting stress test with {i+1} variable parameters out of {n}')

        dim_offline_times = []
        dim_online_times = []
        dim_analytical_times = []
        dim_max_ranks = []
        dim_max_abs_errors = []
        dim_rms_errors = []
        dim_n_samples = []

        for rank in ranks:

            variable_parameters = dict(list(parameter_limits.items())[:i+1])
            fixed_parameters_list = list(parameter_limits.items())[i+1:11]

            fixed_parameters = {k: round((v[0] + v[1])/2, 3) for k, v in fixed_parameters_list}

            pricer = HestonChebyshevTTPricer(
                fixed_params=dict(fixed_parameters),
                variable_params=dict(variable_parameters),
                interpolation_order=interpolation_order,
                tt_rmax=rank,
                tt_max_iter=40,
                tt_eps=1E-3,
                tt_early_stopping_tolerance= 7E-5,
                tt_early_stopping_patience=2
            )

            pricer.offline_phase()
            offline_phase_time = pricer.get_eval_time()
            dim_offline_times.append(offline_phase_time)
            logging.info(f'Offline phase for {list(variable_parameters.keys())} took {offline_phase_time:.2f} seconds')

            dim_max_ranks.append(int(pricer.get_max_tt_rank()))
            dim_n_samples.append(int(pricer.get_n_samples()))

            tt_prices = []
            start_time = time.time()
            for sample in range(n_test):
                test_params = {**{k: test_samples[k][sample] for k in variable_parameters.keys()}, **fixed_parameters}
                price = pricer.price(test_params)
                tt_prices.append(price)
            online_phase_time = (time.time() - start_time) / n_test
            dim_online_times.append(online_phase_time)
            logging.info(f'Online phase for {list(variable_parameters.keys())} took {online_phase_time:.8f} seconds')

            analytical_prices, analytical_time = pricer.fft_pricer(record_time=True, **{**test_samples, **fixed_parameters})
            dim_analytical_times.append(analytical_time)
            logging.info(f'Analytical prices for {list(variable_parameters.keys())} computed')

            max_abs_error = max(abs(a - t) for a, t in zip(analytical_prices, tt_prices))
            dim_max_abs_errors.append(max_abs_error)
            rms_error = (sum((a - t) ** 2 for a, t in zip(analytical_prices, tt_prices)) / n_test) ** 0.5
            dim_rms_errors.append(rms_error)
            logging.info(f'Max absolute error: {max_abs_error:.6f}, RMS error: {rms_error:.6f}')

        offline_times.append(dim_offline_times)
        online_times.append(dim_online_times)
        analytical_times.append(dim_analytical_times)
        max_ranks.append(dim_max_ranks)
        max_abs_errors.append(dim_max_abs_errors)
        rms_errors.append(dim_rms_errors)
        n_samples.append(dim_n_samples)

    output_dict = {
        'ranks': ranks.tolist(),
        'dimensions': dimensions.tolist(),
        'offline_times': offline_times,
        'online_times': online_times,
        'analytical_times': analytical_times,
        'max_ranks': max_ranks,
        'max_abs_errors': max_abs_errors,
        'rms_errors': rms_errors,
        'n_samples': n_samples
    }

    with open('rank_sweep_results.json', 'w') as f:
        json.dump(output_dict, f, indent=4)


if __name__ == '__main__':
    perform_rank_sweep()