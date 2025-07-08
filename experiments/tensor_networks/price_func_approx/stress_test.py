import argparse
import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from heston_tt import build_heston_tt, price_from_tt
from utils.heston_fft import heston_pricer_fft


def perform_stress_test_on_parameter(
        parameter_name: str,
        parameter_limits: tuple,
        error_tolerance: float = 1e-3,
        n_samples: int = 10,
        S: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        v0: float = 0.02,
        kappa: float = 1.0,
        theta: float = 0.02,
        rho: float = -0.7,
        sigma_v: float = 0.1,
        rate: float = 0.05,
        div: float = 0.0
):

    test_cases = {
        'S': [S] * n_samples,
        'K': [K] * n_samples,
        'T': [T] * n_samples,
        'v0': [v0] * n_samples,
        'kappa': [kappa] * n_samples,
        'theta': [theta] * n_samples,
        'rho': [rho] * n_samples,
        'sigma_v': [sigma_v] * n_samples,
        'rate': [rate] * n_samples,
        'div': [div] * n_samples
    }
    parameter_values = np.linspace(parameter_limits[0], parameter_limits[1], n_samples)
    test_cases[parameter_name] = np.linspace(parameter_limits[0], parameter_limits[1], n_samples)

    ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    errors = np.zeros((n_samples, len(ranks)))
    cpu_times = np.zeros((n_samples, len(ranks)))

    for i in tqdm(range(n_samples)):
        params = {key: test_cases[key][i] for key in test_cases}
        analytic_prices = heston_pricer_fft(**params)

        for j, r in enumerate(ranks):
            start_time = time.time()
            tt_heston, info = build_heston_tt(
                base=10,
                basis_size=3,
                tt_tolerance=1e-6,
                tt_rmax=r,
                tt_max_iter=50,
                **params
            )
            tt_prices = price_from_tt(params['K'], params['T'], tt_heston, info)
            errors[i, j] = np.sqrt(np.mean((tt_prices - analytic_prices) ** 2))
            end_time = time.time()
            cpu_times[i, j] = end_time - start_time

    data_for_boxplot = [errors[:, j] for j in range(len(ranks))]

    tolerance = error_tolerance
    worst_case_ranks = np.full(n_samples, np.nan)

    for i in range(n_samples):
        for j, r in enumerate(ranks):
            if errors[i, j] < tolerance:
                worst_case_ranks[i] = r
                break

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), tight_layout=True)

    # rms error vs ranks
    axs[0].boxplot(data_for_boxplot, tick_labels=ranks, showfliers=True)
    axs[0].set_ylabel('RMS Pricing Error')
    axs[0].set_xlabel('Max TT Rank')
    axs[0].set_title(rf'RMS Error vs TT Rank ($\{parameter_name}$ {parameter_limits[0]} to {parameter_limits[1]})')
    axs[0].set_yscale('log')
    axs[0].grid(True, which='both', alpha=0.3)

    # min rank vs parameter values
    axs[1].plot(parameter_values, worst_case_ranks, 'o-', color='orange')
    axs[1].set_xlabel(f'Parameter Value ({parameter_name})')
    axs[1].set_ylabel(f'Min Rank for RMS < {tolerance}')
    axs[1].set_title('Minimum Rank Required per Parameter Draw')
    axs[1].grid(True, alpha=0.3)
    axs[1].set_yticks(ranks)

    # histogram of min ranks
    valid_ranks = worst_case_ranks[~np.isnan(worst_case_ranks)]
    if len(valid_ranks) > 0:
        bins = np.arange(min(ranks) - 0.5, max(ranks) + 1.5, 1)
        axs[2].hist(valid_ranks, bins=bins, rwidth=0.8, align='mid')
        axs[2].set_xticks(ranks)
    axs[2].set_title(f'Distribution of Minimum Rank for RMS < {tolerance}')
    axs[2].set_xlabel('Minimum TT Rank ($r^*$)')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(True, alpha=0.3)

    try:
        plt.savefig(rf'./output/plots/stress_test_{parameter_name}_combined.png', dpi=300)
    except Exception:
        print("Could not save the plot. Ensure the output directory exists.")

    plt.show()

    mean_errors = np.mean(errors, axis=0)
    mean_cpu_times = np.mean(cpu_times, axis=0)

    fig2, ax1 = plt.subplots(figsize=(12, 7))
    fig2.suptitle(rf'Performance Profile ($\{parameter_name}$ {parameter_limits[0]} to {parameter_limits[1]})', fontsize=16)

    color = 'tab:red'
    ax1.set_xlabel('Mean CPU Time (s)', fontsize=12)
    ax1.set_ylabel('Mean RMS Error', color=color, fontsize=12)
    ax1.loglog(mean_cpu_times, mean_errors, 'o-', color=color, label='RMS Error')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Max TT Rank', color=color, fontsize=12)
    ax2.semilogx(mean_cpu_times, ranks, 's--', color=color, label='Max TT Rank', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        plt.savefig(fr'./output/plots/stress_test_{parameter_name}_performance.png', dpi=300)
    except Exception as e:
        print(f"Could not save the performance plot: {e}. Ensure the output directory exists.")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_name", type=str)
    parser.add_argument("parameter_limits", type=float, nargs=2)
    parser.add_argument("--error_tolerance", type=float, default=1e-3)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--S", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=-0.7)
    parser.add_argument("--sigma_v", type=float, default=0.1)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--div", type=float, default=0.0)

    args = parser.parse_args()

    perform_stress_test_on_parameter(
        parameter_name=args.parameter_name,
        parameter_limits=tuple(args.parameter_limits),
        error_tolerance=args.error_tolerance,
        n_samples=args.n_samples,
        S=args.S,
        K=args.K,
        T=args.T,
        v0=args.v0,
        kappa=args.kappa,
        theta=args.theta,
        rho=args.rho,
        sigma_v=args.sigma_v,
        rate=args.rate,
        div=args.div
    )
