"""Stolen from https://github.com/denizkural4/FT-HestonOptionPricing :)"""

import numpy as np
from scipy.ndimage import map_coordinates


def heston_cf(phi, S, T, kappa, rho, volvol, theta, var0, rate, div):
    phi = np.atleast_1d(phi)

    phi_r = phi.reshape(-1, 1)

    xx = np.log(S)

    # Term A: Drift component
    AA = 1j * phi_r * (xx + (rate - div) * T)

    gamma = kappa - rho * volvol * phi_r * 1j
    zeta = -0.5 * (np.power(phi_r, 2) + 1j * phi_r)
    psi = np.power(np.power(gamma, 2) - 2 * np.power(volvol, 2) * zeta, 0.5)
    exp_psi_T = np.exp(-psi * T)
    common_denom = 2 * psi - (psi - gamma) * (1 - exp_psi_T)

    # Term B: Stochastic volatility component
    BB = (2 * zeta * (1 - exp_psi_T) * var0) / common_denom

    # Term C: Mean reversion component
    CCaux = common_denom / (2 * psi)
    CC = (-(kappa * theta) / np.power(volvol, 2)) * (2 * np.log(CCaux) + (psi - gamma) * T)

    return np.exp(AA + BB + CC)


def heston_pricer_fft(S, K, T, sigma_v, kappa, rho, theta, v0, rate, div, batch_size):
    params = [np.atleast_1d(p) for p in [S, K, T, sigma_v, kappa, rho, theta, v0, rate, div]]
    S, K, T, sigma_v, kappa, rho, theta, v0, rate, div = params

    alpha = 1.25
    NN = 4096
    cc = 600
    eta = cc / NN
    Lambda = (2 * np.pi) / (NN * eta)
    bb = (NN * Lambda) / 2

    # Vectorized call to the characteristic function
    jj = np.arange(1, NN + 1)
    phi = eta * (jj - 1)
    new_phi = phi - (alpha + 1) * 1j

    cf = heston_cf(new_phi, S, T, kappa, rho, sigma_v, theta, v0, rate, div)

    denominator = (np.power(alpha, 2) + alpha - np.power(phi, 2) + 1j * phi * (2 * alpha + 1)).reshape(-1, 1)
    modified_cf = (np.exp(-rate * T) * cf) / denominator

    # FFT execution
    delta = np.zeros(NN)
    delta[0] = 1
    simpson = (eta / 3) * (3 + np.power(-1, jj) - delta)

    fft_function = np.exp(1j * bb * phi).reshape(-1, 1) * modified_cf * simpson.reshape(-1, 1)
    payoff = np.real(np.fft.fft(fft_function, axis=0))

    # Interpolation to find the price for the specific strike K
    log_strikes_grid = -bb + Lambda * np.arange(NN)
    call_prices = (np.exp(-alpha * log_strikes_grid).reshape(-1, 1) / np.pi) * payoff

    log_K = np.log(K)
    if log_K.shape[0] == 1 and batch_size > 1:
        log_K = np.repeat(log_K, batch_size)

    coords_d0 = (log_K + bb) / Lambda
    coords_d1 = np.arange(batch_size)
    interp_coords = np.vstack([coords_d0, coords_d1])
    prices = map_coordinates(call_prices, interp_coords, order=1, mode='nearest')

    return prices
