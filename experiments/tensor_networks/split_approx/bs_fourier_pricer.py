import numpy as np
from typing import Optional, Tuple, Union

import tntorch as tn
import torch

from utils.timer import function_timer


class BSFourierPricer:
    """
    Mega class to price basket options under BS. Can price via TT representations of characteristic functions and payoff
    or normally without TT. Uses min-call payoff function as described in the papers which is easy due to closed-form,
    but also includes a (not-very tested) MC based approach to generate the proper weighted payoff. Based on
    https://arxiv.org/pdf/2405.00701 and https://arxiv.org/pdf/2203.02804
    """
    def __init__(
            self,
            d: int = 5,
            T: float = 1.0,
            r: float = 0.05,
            K: float = 100.0,
            N: int = 32,
            eta: float = 0.3,
            s0_range: Tuple[float, float] = (90.0, 120.0),
            vol_range: Tuple[float, float] = (0.15, 0.25),
            phi_tt_params: Optional[dict] = None,
            vhat_tt_params: Optional[dict] = None,
            random_state: Optional[int] = None,
    ):
        self.d, self.T, self.r, self.K, self.N, self.eta = d, T, r, K, N, eta
        self.rng = np.random.default_rng(random_state)
        np.random.seed(random_state)

        self.alpha = 5 / d

        self.s0 = self.rng.uniform(*s0_range, size=d)
        self.vol = self.rng.uniform(*vol_range, size=d)

        self.corr_matrix = self._generate_correlation_matrix()
        self._compute_parameters()

        default_tt_params = {
            'eps': 1e-5,
            'rmax': 40,
            'max_iter': 100,
            'early_stopping_patience': 3,
            'early_stopping_tolerance': 1e-6,
            'return_info': True
        }

        self.phi_tt = None
        self.phi_tt_info = None
        self.phi_tt_params = phi_tt_params if phi_tt_params is not None else default_tt_params
        self.vhat_tt = None
        self.vhat_tt_info = None
        self.vhat_tt_params = vhat_tt_params if vhat_tt_params is not None else default_tt_params

    def _generate_correlation_matrix(self) -> np.ndarray:
        A = self.rng.standard_normal((self.d, self.d))
        cov = A @ A.T
        D = np.diag(1.0 / np.sqrt(np.diag(cov)))
        return D @ cov @ D

    def _compute_parameters(self) -> None:
        self.mu = np.log(self.s0) + (self.r - 0.5 * self.vol**2) * self.T
        self.Sigma = np.diag(self.vol) @ self.corr_matrix @ np.diag(self.vol) * self.T

    def _make_Z(self, omega):
        N = omega.shape[0]
        # allocate (N, N, …, N, d)
        Z = np.empty((N,)*self.d + (self.d,), dtype=omega.dtype)
        # prepare shape template
        shape = [1]*self.d
        for i in range(self.d):
            view_shape = shape.copy()
            view_shape[i] = N
            Z[..., i] = omega.reshape(view_shape)
        return Z

    def _initialise_parameters(self) -> None:
        self.alpha = 5 / self.d
        self.s0 = self.rng.uniform(90.0, 120.0, size=self.d)
        self.vol = self.rng.uniform(0.15, 0.25, size=self.d)
        self.corr_matrix = self._generate_correlation_matrix()

    def generate_grid(self) -> np.ndarray:
        j = np.arange(-self.N // 2, self.N // 2)
        omega = self.eta * j + 1j * self.alpha
        return self._make_Z(omega)

    def adjust_parameters(
            self,
            d: Optional[int] = None,
            T: Optional[float] = None,
            r: Optional[float] = None,
            K: Optional[float] = None,
            eta: Optional[float] = None,
            s0_range: Optional[Tuple[float, float]] = None,
            vol_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        if d is not None:
            self.d = d
            self._initialise_parameters()
        if T is not None:
            self.T = T
        if r is not None:
            self.r = r
        if K is not None:
            self.K = K
        if eta is not None:
            self.eta = eta
        if s0_range is not None:
            self.s0 = self.rng.uniform(*s0_range, size=self.d)
        if vol_range is not None:
            self.vol = self.rng.uniform(*vol_range, size=self.d)

        self._compute_parameters()

    def compute_vhat_mc(z_grid, mu, Sigma, d, K, M=10000):
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

    def compute_vhat_min_call_tt(self, z: np.ndarray) -> torch.tensor:
        s = np.sum(z, axis=0)
        numerator = -self.K**(1 + 1j * s)
        denominator = ((-1)**self.d) * (1 + 1j * s) * np.prod(1j * z, axis=0)
        return torch.from_numpy(numerator / denominator)  # annoying typing due to tntorch wanting a tensor

    def compute_vhat_min_call(self, z: np.ndarray) -> np.ndarray:
        s = z.sum(axis=-1)
        numerator = -self.K**(1 + 1j * s)
        denominator = ((-1)**self.d) * (1 + 1j * s) * np.prod(1j * z, axis=-1)
        return numerator / denominator  # no typing issues as this approach doesn't use tt-cross

    def compute_phi_tt(self, z: np.ndarray) -> torch.tensor:
        z_dot_mu = np.dot(self.mu, z)
        Sigma_z = self.Sigma @ z
        quad = np.sum(z * Sigma_z, axis=0)
        return torch.from_numpy(np.exp(1j * z_dot_mu - 0.5 * quad))  # annoying typing due to tntorch wanting a tensor

    def compute_phi(self, z: np.ndarray) -> np.ndarray:
        z_dot_mu = np.tensordot(z, self.mu, axes=([-1], [0]))
        Sigma_z = np.tensordot(z, self.Sigma, axes=([-1], [0]))
        quad = np.sum(z * Sigma_z, axis=-1)
        return np.exp(1j * z_dot_mu - 0.5 * quad)  # no typing issues as this approach doesn't use tt-cross

    def phi_entry(self, *indices: int) -> np.ndarray:
        j = np.array(indices) - self.N // 2
        z = self.eta * j + 1j * self.alpha
        return self.compute_phi_tt(-z)

    def vhat_entry(self, *indices: int) -> np.ndarray:
        j = np.array(indices) - self.N // 2
        z = self.eta * j + 1j * self.alpha
        return self.compute_vhat_min_call_tt(z)

    def joint_entry(self, *indices: int) -> np.ndarray:
        j = np.array(indices) - self.N // 2
        z = self.eta * j + 1j * self.alpha
        return self.compute_phi_tt(-z) * self.compute_vhat_min_call_tt(z)

    @function_timer
    def run_phi_tt_cross(self):
        phi_tt, phi_info = tn.cross(
            function=self.phi_entry,
            domain=[self.N] * self.d,
            **self.phi_tt_params
        )
        self.phi_tt = phi_tt
        self.phi_tt_info = phi_info
        return phi_info

    @function_timer
    def run_vhat_tt_cross(self):
        vhat_tt, vhat_info = tn.cross(
            function=self.vhat_entry,
            domain=[self.N] * self.d,
            **self.vhat_tt_params
        )
        self.vhat_tt = vhat_tt
        self.vhat_tt_info = vhat_info
        return vhat_info

    @function_timer
    def price_from_tt(self, return_imaginary: bool = False) -> Union[float, complex]:
        if not (self.phi_tt and self.vhat_tt):
            raise ValueError("TT representations for phi and vhat must be computed first.")
        price = np.exp(-self.r * self.T) * (self.eta**self.d) / (2 * np.pi)**self.d * np.dot(self.phi_tt, self.vhat_tt).sum()

        if return_imaginary:
            return price.item()
        else:
            return price.real.item()

    @function_timer
    def price_from_fourier(self, return_imaginary: bool = False) -> Union[float, complex]:
        z = self.generate_grid()

        integrand = self.compute_phi(-z) * self.compute_vhat_min_call(z)
        price = np.exp(-self.r * self.T) * (self.eta**self.d) / (2 * np.pi)**self.d * integrand.sum()

        if return_imaginary:
            return price.item()
        else:
            return price.real.item()
