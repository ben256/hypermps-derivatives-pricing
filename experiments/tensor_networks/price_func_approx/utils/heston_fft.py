"""Stolen from https://github.com/denizkural4/FT-HestonOptionPricing :)"""

import numpy as np


def _to_np_array(*args):
    """Helper function to convert scalar inputs to numpy arrays."""
    if len(args) == 1:
        return np.atleast_1d(args[0])
    return [np.atleast_1d(arg) for arg in args]


def HestonCf(phi, S, T, kappa, rho, volvol, theta, var0, rate, div):
    params = [np.atleast_1d(p) for p in [S, T, kappa, rho, volvol, theta, var0, rate, div]]
    S, T, kappa, rho, volvol, theta, var0, rate, div = params
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


def heston_pricer_fft(S, K, T, sigma_v, kappa, rho, theta, v0, rate, div):
    S, K = _to_np_array(S, K)

    alpha = 1.25
    NN = 4096
    cc = 600
    eta = cc / NN
    Lambda = (2 * np.pi) / (NN * eta)
    bb = (NN * Lambda) / 2

    # Vectorized call to the characteristic function
    jj = np.arange(1, NN + 1)
    phi = eta * (jj - 1)
    NewPhi = phi - (alpha + 1) * 1j

    CF = HestonCf(NewPhi, S, T, kappa, rho, sigma_v, theta, v0, rate, div)
    phi_r = phi.reshape(-1, 1)

    ModCF = (np.exp(-rate * T) * CF) / (np.power(alpha, 2) + alpha - np.power(phi_r, 2) + 1j * phi_r * (2 * alpha + 1))

    # FFT execution
    delta = np.zeros(NN)
    delta[0] = 1
    Simpson = (eta / 3) * (3 + np.power(-1j, jj) - delta)
    Simpson_r = Simpson.reshape(-1, 1)

    FuncFFT = np.exp(1j * bb * phi_r) * ModCF * Simpson_r
    Payoff = np.real(np.fft.fft(FuncFFT, axis=0))

    # Interpolation to find the price for the specific strike K
    ku = -bb + Lambda * (jj - 1)
    ku_r = ku.reshape(-1, 1)
    CallPrices = (np.exp(-alpha * ku_r) / np.pi) * Payoff

    interp_positions = ((np.log(K) + bb) / Lambda) + 1
    prices = np.zeros(S.size)
    for i in range(S.size):
        prices[i] = np.interp(interp_positions[i], jj, CallPrices[:, i])

    return prices
