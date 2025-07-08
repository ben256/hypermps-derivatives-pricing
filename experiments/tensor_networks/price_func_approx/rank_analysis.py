import numpy as np
import matplotlib.pyplot as plt

from legacy.heston_tt import approximate_heston_price_with_tt
from utils.heston_fft import heston_pricer_fft


def plot_rank_analysis():

    S_limit = (90.0, 110.0)
    K_limit = (90.0, 110.0)
    T_limit = (0.0, 1.0)
    v0_limit = (0.01, 0.04)

    kappa = 1.0
    theta = 0.02
    rho = -0.7
    sigma_v = 0.1
    r = 0.05

    grid_ranges = [np.linspace(lim[0], lim[1], 10) for lim in (S_limit, K_limit, T_limit, v0_limit)]
    grid_coords = np.stack(np.meshgrid(*grid_ranges, indexing='ij'), axis=-1)
    S_test, K_test, T_test, v0_test = grid_coords.reshape(-1, 4).T
    test_values = (S_test, K_test, T_test, v0_test)

    # analytical_price = heston_pricer_fft(S_test, K_test, T_test, sigma_v, kappa, rho, theta, v0_test, r, 0.0)
    analytical_prices = np.load('output/tensor_trains/heston_tt_analytical_prices.npy')

    ranks_to_test = [2, 4, 8, 16, 32]
    all_errors = []
    mse_errors = []
    approx_error = []

    for test_rank in ranks_to_test:
        tt_heston, tt_cross_info, tt_prices = approximate_heston_price_with_tt(
            tt_tolerance=1e-4,
            tt_rmax=test_rank,
            tt_max_iter=50,
            test_values=test_values,
            S_limit=S_limit,
            K_limit=K_limit,
            T_limit=T_limit,
            v0_limit=v0_limit,
            kappa=kappa,
            theta=theta,
            rho= rho,
            sigma_v=sigma_v,
            r=r,
        )
        errors = np.abs(analytical_prices - tt_prices)
        all_errors.append(errors)
        mse_errors.append(np.mean(errors**2))
        approx_error.append(tt_cross_info['val_eps'])
        print(f"Rank: {test_rank}, Mean Absolute Error: {np.mean(errors)}")

    fig, ax = plt.subplots(2, 1, figsize=(9, 15), tight_layout=True)

    ax[0].boxplot(all_errors, labels=[str(r) for r in ranks_to_test])

    ax[0].set_title('Box Plot of TT Approximation Error by Rank')
    ax[0].set_xlabel('TT Rank')
    ax[0].set_ylabel('Absolute Error')
    ax[0].set_yscale('log')
    ax[0].grid(visible=True, which='major')
    ax[0].grid(visible=True, which='minor', linestyle='-', alpha=0.3)
    ax[0].minorticks_on()
    ax[0].legend()

    ax[1].plot(ranks_to_test, mse_errors, marker='o', linestyle='-', color='b', label='Mean Squared Error')
    ax[1].plot(ranks_to_test, approx_error, marker='x', linestyle='--', color='r', label='Approximation Error')

    ax[1].set_title('Mean Squared Error vs TT Rank')
    ax[1].set_xlabel('TT Rank')
    ax[1].set_ylabel('Mean Squared Error')
    ax[1].set_yscale('log')
    ax[1].grid(visible=True, which='major')
    ax[1].grid(visible=True, which='minor', linestyle='-', alpha=0.3)
    ax[1].minorticks_on()
    ax[1].legend()

    plt.show()
    plt.close()


if __name__ == '__main__':
    plot_rank_analysis()