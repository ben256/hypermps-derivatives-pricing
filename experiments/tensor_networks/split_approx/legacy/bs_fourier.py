import random
import time
from typing import List

import numpy as np
import scipy


np.random.seed(42)


class BasketOptionPricer:
    def __init__(
            self,
            d: int = 5
    ):
        ...


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
    s = z_grid.sum(axis=-1)
    numerator = -K**(1 + 1j * s)
    denominator = ((-1)**d) * (1 + 1j * s) * np.prod(1j * z_grid, axis=-1)
    return numerator / denominator


def vhat_mc(z_grid, mu, Sigma, d, K, M=10000):

    X = np.random.multivariate_normal(mu, Sigma, size=M)

    S_T = np.exp(X)
    basket = S_T.mean(axis=1)
    payoff = np.maximum(basket - K, 0)

    original_shape = z_grid.shape
    z_flat = z_grid.reshape(-1, d)

    exponents = X @ z_flat.T
    integrand = np.exp(1j * exponents) * payoff[:, None]
    vhat_flat = integrand.mean(axis=0)

    return vhat_flat.reshape(original_shape[:-1])


def phi(z_grid, mu, Sigma):
    z_dot_mu = np.tensordot(z_grid, mu, axes=([-1], [0]))
    Sigma_z = np.tensordot(z_grid, Sigma, axes=([-1], [0]))
    quad = np.sum(z_grid * Sigma_z, axis=-1)
    return np.exp(1j * z_dot_mu - 0.5 * quad)


def make_Z(omega, d):
    N = omega.shape[0]
    Z = np.empty((N,)*d + (d,), dtype=omega.dtype)
    shape = [1]*d
    for i in range(d):
        view_shape = shape.copy()
        view_shape[i] = N
        Z[..., i] = omega.reshape(view_shape)
    return Z


def fourier_pricing(
        d: int = 5,
        T: float = 1.0,
        r: float = 0.05,
        K: float = 80.0,
        N: int = 42,
        eta: float = 0.3,
        s0_range: tuple[float, float] = (90.0, 120.0),
        vol_range: tuple[float, float] = (0.15, 0.25),
):
    """
    d: underlying assets
    T: time horizon
    r: risk-free rate
    K: strike price
    N: number of Fourier grid points
    eta: spacing of Fourier grid points
    """

    # Fourier Grid
    alpha = 5/d
    j = np.arange(-N // 2, N // 2)
    omega = eta * j + 1j * alpha
    # Z = make_Z(omega, d)

    # Model params
    s0 = np.random.uniform(s0_range[0], s0_range[1], size=d)
    vol = np.random.uniform(vol_range[0], vol_range[1], size=d)
    R = generate_correlated_R(d)  # Correlation Matrix
    mu = np.log(s0) + (r - 0.5 * vol**2) * T
    Sigma = np.diag(vol) @ R @ np.diag(vol) * T

    grids = np.meshgrid(*([omega] * d), indexing='ij')
    Z = np.stack(grids, axis=-1)

    print('Calculating Phi and Vhat...')
    start_time = time.time()
    Phi = phi(-Z, mu, Sigma)
    mid_time = time.time()
    print(f'Time to calculate Phi: {mid_time - start_time:.4f} seconds')
    # Vhat = vhat_mc(Z, mu, Sigma, d, K)
    Vhat = vhat_min_call(Z, K, d)
    time_span = time.time() - mid_time
    print(f'Time to calculate Vhat: {time_span:.4f} seconds')

    start_time = time.time()
    integrand = Phi * Vhat
    price = np.exp(-r * T) * (eta**d) / (2 * np.pi)**d * integrand.sum()
    time_span = time.time() - start_time
    print(f'Time to calculate integral: {time_span:.4f} seconds')

    print(f"Basket Option Price: {price.real}")
    return price.real


fourier_pricing()
