import logging
import time

import numpy as np
import torch
from scipy.fft import dct

from utils.heston_fft import heston_pricer_fft

import tntorch as tn


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


class HestonChebyshevTTPricer:
    """
    Chebyshev based interpolation pricer using Heston FFT pricer and TT-cross to generate TT representation of pricing
    grid. Based on this paper https://arxiv.org/pdf/1902.04367
    """
    def __init__(
        self,
        fixed_params: dict,
        variable_params: dict,
        interpolation_order: int = 6,
        tt_rmax: int = 30,
        tt_max_iter: int = 20,
        tt_eps: float = 1E-6,
        tt_early_stopping_patience: int = 2,
        tt_early_stopping_tolerance: float = 1e-7,
    ):
        self.fixed_params = fixed_params
        self.variable_params = variable_params
        self.interpolation_order = interpolation_order
        self.tt_rmax = tt_rmax
        self.tt_max_iter = tt_max_iter
        self.tt_eps = tt_eps
        self.tt_early_stopping_patience = tt_early_stopping_patience
        self.tt_early_stopping_tolerance = tt_early_stopping_tolerance

        self.d = len(variable_params)
        self.param_names = list(variable_params.keys())
        self.coefficients = None
        self.offline_info = None


    def _map_to_unit_interval(
        self,
        param_name: str,
        param_value: float,
    ):
        param_min, param_max = self.variable_params[param_name]
        return -1.0 + 2.0 * (param_value - param_min) / (param_max - param_min)


    def _map_from_unit_interval(
        self,
        param_name: str,
        param_value: float,
    ):
        param_min, param_max = self.variable_params[param_name]
        return param_min + 0.5 * (param_value + 1) * (param_max - param_min)


    def _compute_coefficient_tensor(
        self,
        P: tn.tensor,
    ):
        C = P.clone()
        for m in range(self.d):
            n = P.shape[m] - 1

            if n == 0:
                continue

            F_n = dct(np.eye(n + 1), type=1) / n
            F_n[:, 0] /= 2
            F_n[:, -1] /= 2
            F_n = torch.from_numpy(F_n)

            core = C.cores[m]
            new_core = torch.einsum('ijk,lj->ilk', core, F_n)
            C.cores[m] = new_core

        return C


    def _chebyshev_polynomials(self, x, n):
        T = np.zeros(n + 1)
        T[0] = 1.0
        if n > 0:
            T[1] = x

        for i in range(1, n):
            T[i+1] = 2 * x * T[i] - T[i-1]

        return torch.from_numpy(T)

    @staticmethod
    def fft_pricer(record_time=False, **params):
        if record_time:
            start_time = time.time()
        batch_size = params.pop('batch_size', 1)
        prices = heston_pricer_fft(**params, batch_size=batch_size)

        if record_time:
            inference_time = (time.time() - start_time) / batch_size
            return prices, inference_time

        else:
            return prices


    def get_max_tt_rank(self):
        return self.offline_info['Rs'].max()


    def get_n_samples(self):
        return self.offline_info['nsamples']


    def get_eval_time(self):
        return self.offline_info['total_time']


    def offline_phase(self):

        nodes = [np.cos(np.pi * np.arange(self.interpolation_order + 1) / self.interpolation_order) for i in range(self.d)]

        def f(*multi_index):
            variable_params_values = {
                self.param_names[i]: self._map_from_unit_interval(
                    self.param_names[i], np.take(nodes[i], multi_index[i])
                )
                for i in range(self.d)
            }
            batch_size = len(multi_index[0])
            all_params = {**variable_params_values, **self.fixed_params, 'batch_size': batch_size}
            prices = heston_pricer_fft(**all_params)
            return torch.from_numpy(prices)

        price_tensor, info = tn.cross(
            function=f,
            domain=(self.interpolation_order + 1,)*self.d,
            rmax=self.tt_rmax,
            max_iter=self.tt_max_iter,
            eps=self.tt_eps,
            early_stopping_patience=self.tt_early_stopping_patience,
            early_stopping_tolerance=self.tt_early_stopping_tolerance,
            return_info=True
        )

        self.offline_info = info
        self.coefficients = self._compute_coefficient_tensor(price_tensor)


    def price(self, test_params: dict):
        T_p_cores = []
        for i in range(self.d):
            param_name = self.param_names[i]
            param_value = test_params[param_name]
            mapped_value = self._map_to_unit_interval(param_name, param_value)
            poly_evals = self._chebyshev_polynomials(mapped_value, self.interpolation_order)
            core = poly_evals.reshape(1, self.interpolation_order + 1, 1)
            T_p_cores.append(core)

        T_p = tn.Tensor(T_p_cores)

        interpolated_price = tn.dot(self.coefficients, T_p)
        return interpolated_price.item()


# Probably use rank_sweep.py or stress_test.py instead
# if __name__ == '__main__':
#     params_to_interpolate = {
#         'T': [0.1, 2.0],
#         'v0': [0.01, 0.2],
#         'sigma_v': [0.1, 1.0],
#         'rho': [-0.9, -0.1],
#         'kappa': [0.5, 5.0],
#         # 'S': [80.0, 120.0],
#         # 'K': [80.0, 120.0],
#         # 'theta': [0.01, 0.1],
#         # 'rate': [0.01, 0.1],
#         'div': [0.0, 0.05]
#     }
#
#     fixed_parameters = {
#         'S': 100,
#         'K': 100,
#         'theta': 0.04,
#         'rate': 0.05,
#         # 'div': 0.0,
#         # 'kappa': 2.0,
#         # 'v0': 0.04,
#         # 'sigma_v': 0.3,
#         # 'rho': -0.7
#     }
#
#     d = len(params_to_interpolate)
#     interpolation_order = 10
#
#     chebyshev_pricer = HestonChebyshevTTPricer(fixed_parameters, params_to_interpolate, interpolation_order)
#     chebyshev_pricer.offline_phase()
#     test_params = {
#         'T': 1.0,
#         'v0': 0.04,
#         'sigma_v': 0.3,
#         'rho': -0.7,
#         'kappa': 2.0,
#         'S': 100.0,
#         'K': 100.0,
#         'theta': 0.04,
#         'rate': 0.05,
#         'div': 0.0
#     }
#
#     tt_price = chebyshev_pricer.price(test_params)
#     print(f"Chebyshev TT Price: {tt_price}")
#     fft_price = heston_pricer_fft(**test_params, batch_size=1)
#     print(f"FFT Price: {fft_price}")
#     error = np.abs(tt_price - fft_price[0])
#     print(f"Error: {error:.6f}")
