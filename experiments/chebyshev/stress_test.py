import logging
import time
import random

import matplotlib.pyplot as plt
import numpy as np

from heston_chebyshev_tt import HestonChebyshevTTPricer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def perform_stress_test():
    """
    Iterate through the number of variable parameters (dimensions), running TT-cross each cycle and measuring
    performance. This code is used in notebook to produce various graphs.
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

    interpolation_order = 10
    n = len(parameter_limits.keys())
    dimensions = np.arange(1, n + 1)

    offline_times = []
    online_times = []
    max_ranks = []
    max_abs_errors = []
    rms_errors = []
    n_samples = []

    for i in range(n):

        logging.info(f'Starting stress test with {i+1} variable parameters out of {n}')

        variable_parameters = dict(list(parameter_limits.items())[:i+1])
        fixed_parameters_list = list(parameter_limits.items())[i+1:11]

        fixed_parameters = {k: round((v[0] + v[1])/2, 3) for k, v in fixed_parameters_list}

        pricer = HestonChebyshevTTPricer(
            fixed_params=dict(fixed_parameters),
            variable_params=dict(variable_parameters),
            interpolation_order=interpolation_order,
            tt_rmax=250,
            tt_max_iter=30,
            tt_eps=1E-3,
            tt_early_stopping_tolerance= 1E-4,
            tt_early_stopping_patience=2
        )

        pricer.offline_phase()
        offline_phase_time = pricer.get_eval_time()
        offline_times.append(offline_phase_time)
        logging.info(f'Offline phase for {list(variable_parameters.keys())} took {offline_phase_time:.2f} seconds')

        max_ranks.append(pricer.get_max_tt_rank())
        n_samples.append(pricer.get_n_samples())

        tt_prices = []
        start_time = time.time()
        for sample in range(n_test):
            test_params = {**{k: test_samples[k][sample] for k in variable_parameters.keys()}, **fixed_parameters}
            price = pricer.price(test_params)
            tt_prices.append(price)
        online_phase_time = (time.time() - start_time) / n_test
        online_times.append(online_phase_time)
        logging.info(f'Online phase for {list(variable_parameters.keys())} took {online_phase_time:.8f} seconds')

        analytical_prices = pricer.fft_pricer(**{**test_samples, **fixed_parameters})
        logging.info(f'Analytical prices for {list(variable_parameters.keys())} computed')

        max_abs_error = max(abs(a - t) for a, t in zip(analytical_prices, tt_prices))
        max_abs_errors.append(max_abs_error)
        rms_error = (sum((a - t) ** 2 for a, t in zip(analytical_prices, tt_prices)) / n_test) ** 0.5
        rms_errors.append(rms_error)
        logging.info(f'Max absolute error: {max_abs_error:.6f}, RMS error: {rms_error:.6f}')


    fig, ax = plt.subplots(figsize=(12, 8))

    # fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    #
    # fig.suptitle('Chebyshev-TT Pricer Stress Test Results', fontsize=18)
    #
    # # Plot 1: Cost and Complexity vs. Dimensionality
    # ax1 = ax[0]
    # ax1_twin = ax1.twinx()
    # ax1.set_title('Computational Cost & Model Complexity vs. Dimensionality', fontsize=14)
    # ax1.set_xlabel('Number of Variable Parameters (d)')
    # ax1.set_ylabel('CPU Time (s) [Log Scale]', color='tab:red')
    # ax1.plot(dimensions, offline_times, 'o-', color='firebrick', label='Offline (Training) Time')
    # ax1.plot(dimensions, online_times, 's--', color='salmon', label='Online (Inference) Time per Sample')
    # ax1.set_yscale('log')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    # ax1.grid(True, which="both", alpha=0.3)
    #
    # ax1_twin.set_ylabel('Max TT Rank', color='tab:blue')
    # ax1_twin.plot(dimensions, max_ranks, '^-', color='steelblue', label='Max TT Rank')
    # ax1_twin.tick_params(axis='y', labelcolor='tab:blue')
    #
    # # Combine legends for the twin axes plot
    # lines, labels = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax1_twin.get_legend_handles_labels()
    # ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    #
    # # Plot 2: Accuracy Degradation vs. Dimensionality
    # ax2 = ax[1]
    # ax2.set_title('Accuracy Degradation vs. Dimensionality', fontsize=14)
    # ax2.set_xlabel('Number of Variable Parameters (d)')
    # ax2.set_ylabel('Error (log)')
    # ax2.plot(dimensions, rms_errors, 'o-', color='darkgreen', label='Root Mean Square Error')
    # ax2.plot(dimensions, max_abs_errors, 's--', color='limegreen', label='Max Absolute Error')
    # ax2.set_yscale('log')
    # ax2.grid(True, which="both", alpha=0.3)
    # ax2.legend()
    #
    # # Plot 3: Offline Time Explosion (Linear Scale)
    # ax3 = ax[2]
    # ax3.set_title('Offline Phase Time vs. Dimensionality (Linear Scale)', fontsize=14)
    # ax3.set_xlabel('Number of Variable Parameters (d)')
    # ax3.set_ylabel('CPU Time (s)')
    # ax3.plot(dimensions, offline_times, 'o-', color='purple', label='Offline (Training) Time')
    # ax3.grid(True, which="both", alpha=0.3)
    # ax3.legend()
    #
    # fig.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('output/stress_test_results.png', dpi=300)
    # logging.info("Stress test complete. Results saved to stress_test_results.png")
    # plt.show()


if __name__ == '__main__':
    perform_stress_test()


