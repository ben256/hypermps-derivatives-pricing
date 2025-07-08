import random
import time
from typing import List

import numpy as np
import tntorch as tn
import scipy
import torch

np.random.seed(42)


def generate_basket_path(
        s0: np.ndarray,
        vol: np.ndarray,
        R: np.ndarray,
        d: int = 5,
        r: float = 0.05,
        T: float = 1.0,
):
    """
    N: number of underlying assets
    r: risk-free rate
    T: time horizon
    """

    L = np.linalg.cholesky(R)  # Lower triangular matrix

    Z = np.random.randn(d)  # Independent normals
    dW = np.sqrt(T) * (L @ Z)

    drift = (r - 0.5 * vol**2) * T
    diffusion = vol * dW

    logS_T = np.log(s0) + drift + diffusion
    S_T = np.exp(logS_T)

    return S_T


def generate_correlated_R(N):
    A = np.random.randn(N, N)
    vol = A @ A.T
    D = np.diag(1 / np.sqrt(np.diag(vol)))
    return D @ vol @ D


def vhat_min_call(z_grid, K, d):
    s = np.sum(z_grid, axis=0)
    numerator = -K**(1 + 1j * s)
    denominator = ((-1)**d) * (1 + 1j * s) * np.prod(1j * z_grid, axis=0)
    return numerator / denominator


def phi(z_grid, mu, Sigma):
    z_dot_mu = np.dot(mu, z_grid)
    Sigma_z = Sigma @ z_grid
    quad = np.sum(z_grid * Sigma_z, axis=0)
    return np.exp(1j * z_dot_mu - 0.5 * quad)


def fourier_pricing(
        d: int = 5,
        T: float = 1.0,
        r: float = 0.05,
        K: float = 50.0,
        N: int = 50,
        eta: float = 0.2,
        s0_range: tuple[float, float] = (90.0, 120.0),
        vol_range: tuple[float, float] = (0.15, 0.25),
):
    """
    d: underlying assets
    T: time horizon
    r: risk-free rate
    K: strike price
    N: number of Fourier coefficients
    eta: spacing of Fourier grid
    """

    # Fourier Grid
    alpha = 5/d

    # Model params
    s0 = np.random.uniform(s0_range[0], s0_range[1], size=d)
    vol = np.random.uniform(vol_range[0], vol_range[1], size=d)
    R = generate_correlated_R(d)  # Correlation Matrix
    mu = np.log(s0) + (r - 0.5 * vol**2) * T
    Sigma = np.diag(vol) @ R @ np.diag(vol) * T

    def phi_entry(*indices):
        j = np.array(indices) - N // 2
        z = eta * j + 1j * alpha

        return torch.from_numpy(phi(-z, mu, Sigma))

    def vhat_entry(*indices):
        j = np.array(indices) - N // 2
        z = eta * j + 1j * alpha

        return torch.from_numpy(vhat_min_call(z, K, d))

    # TT-cross
    phi_tt, phi_info = tn.cross(
        function=phi_entry,
        domain=[N] * d,
        eps=1e-5,
        rmax=40,
        max_iter=100,
        early_stopping_patience=3,
        return_info=True,
    )

    vhat_tt, vhat_info = tn.cross(
        function=vhat_entry,
        domain=[N] * d,
        eps=1e-5,
        rmax=20,
        max_iter=100,
        early_stopping_patience=3,
        return_info=True,
    )

    price = np.exp(-r * T) * (eta**d) / (2 * np.pi)**d * np.dot(phi_tt, vhat_tt).sum()
    test = price.item()
    test2 = price.real.item()

    print("Price:", price.real)
    return price.real


fourier_pricing()
