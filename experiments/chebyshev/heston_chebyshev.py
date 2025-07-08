import logging
import time

import numpy as np
from scipy.fft import dctn, dct

from utils.heston_fft import heston_pricer_fft

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


class HestonChebyshevPricer:
    """
    Basic Chebyshev interpolation based pricer using Heston FFT model to generate the price grid. Though looking back
    I think the online phase is actually wrong since it is just looking up values rather than interpolating... the TT
    based pricer fixes this by actually interpolating. Based on https://arxiv.org/pdf/1505.04648
    """
    def __init__(
            self,
            params_config: dict,
            fixed_params: dict,
            interpolation_orders: list,
    ):
        self.params_config = params_config
        self.fixed_params = fixed_params
        self.interpolation_orders = interpolation_orders
        self.param_names = list(params_config.keys())
        self.d = len(self.param_names)
        self.coeffs = None


    def map_from_unit_interval(
            self,
            param_name: str,
            param_value: float,
    ):
        param_min, param_max = self.params_config[param_name]
        return param_min + (param_max - param_min) * (param_value + 1) / 2


    def map_to_unit_interval(
            self,
            param_name: str,
            param_value: float,
    ):
        param_min, param_max = self.params_config[param_name]
        return 2 * (param_value - param_min) / (param_max - param_min) - 1


    def offline_phase(self):
        logging.info('Starting offline phase.')
        start_time = time.time()

        chebyshev_nodes = [np.cos(np.pi * np.arange(self.interpolation_orders[i] + 1) / self.interpolation_orders[i]) for i in range(self.d)]  # just after eq 2.10 in paper, shown in fig 2.2
        price_tensor = np.zeros([i + 1 for i in self.interpolation_orders])

        mapped_chebyshev_nodes = [self.map_from_unit_interval(self.param_names[i], chebyshev_nodes[i]) for i in range(self.d)]
        mapped_param_grid = np.meshgrid(*mapped_chebyshev_nodes, indexing='ij')

        iterator = np.nditer(price_tensor, flags=['multi_index'], op_flags=['writeonly'])
        with iterator:
            while not iterator.finished:
                index = iterator.multi_index
                current_params = self.fixed_params.copy()

                for i in range(self.d):
                    param_name = self.param_names[i]
                    current_params[param_name] = mapped_param_grid[i][index]

                price = heston_pricer_fft(
                    S=current_params['S'],
                    K=current_params['K'],
                    T=current_params['T'],
                    sigma_v=current_params['sigma_v'],
                    kappa=current_params['kappa'],
                    rho=current_params['rho'],
                    theta=current_params['theta'],
                    v0=current_params['v0'],
                    rate=current_params['rate'],
                    div=current_params['div']
                )
                iterator[0] = price[0]
                iterator.iternext()

        logging.info('Generated price tensor.')
        coeffs_tensor = price_tensor.copy().astype(float)
        for axis in range(price_tensor.ndim):

            N = coeffs_tensor.shape[axis]
            if N < 2:
                continue

            # kinda equivalent to equation 2.9 but using scipy dct instead of maunally, though means that scaling is needed
            coeffs_tensor = dct(coeffs_tensor, type=1, axis=axis, norm=None)

            n = N - 1
            w = np.ones(N) / n
            w[0] = 1 / (2 * n)
            w[-1] = 1 / (2 * n)

            shape = [1] * coeffs_tensor.ndim
            shape[axis] = N
            w = w.reshape(shape)
            coeffs_tensor = coeffs_tensor * w

        self.coeffs = coeffs_tensor
        logging.info('Computed Chebyshev coefficients.')
        end_time = time.time()
        logging.info(f'Offline phase completed in {end_time - start_time:.2f} seconds.')


    def online_phase(
            self,
            params: dict,
    ):
        unit_params = np.array([self.map_to_unit_interval(p_name, params[p_name]) for p_name in self.param_names])

        chebyshev_evaluations = []
        for i in range(self.d):
            # chebyshevy polynomials for each param
            n_i = self.interpolation_orders[i]
            j = np.arange(n_i + 1)
            chebyshev_evaluations.append(np.cos(j * np.arccos(unit_params[i])))

        einsum_str_indices = ''.join(chr(ord('i')+j) for j in range(self.d))
        einsum_str = ','.join(einsum_str_indices) + ',' + einsum_str_indices + '->'

        return np.einsum(einsum_str, *chebyshev_evaluations, self.coeffs)  # eq 2.7 in le paper


if __name__ == '__main__':
    params_to_interpolate = {
        'T': [0.1, 2.0],
        'v0': [0.01, 0.2],
        'sigma_v': [0.1, 1.0],
        'rho': [-0.9, -0.1],
        'kappa': [0.5, 5.0]
    }

    fixed_parameters = {
        'S': 100,
        'K': 100,
        'theta': 0.04,
        'rate': 0.05,
        'div': 0.0
    }

    d = len(params_to_interpolate)
    interpolation_orders = [4] * d

    chebyshev_pricer = HestonChebyshevPricer(params_to_interpolate, fixed_parameters, interpolation_orders)
    chebyshev_pricer.offline_phase()

    test_params = {
        'T': 1.0,
        'v0': 0.04,
        'sigma_v': 0.3,
        'rho': -0.7,
        'kappa': 2.0
    }

    start_time = time.time()
    chebyshev_price = chebyshev_pricer.online_phase(test_params)
    end_time = time.time()
    logging.info(f"Chebyshev Price: {chebyshev_price:.6f} (calculated in {(end_time - start_time):.8f} seconds)")

    full_test_params = {**fixed_parameters, **test_params}

    start_time = time.time()
    fft_price = heston_pricer_fft(
        S=full_test_params['S'],
        K=full_test_params['K'],
        T=full_test_params['T'],
        sigma_v=full_test_params['sigma_v'],
        kappa=full_test_params['kappa'],
        rho=full_test_params['rho'],
        theta=full_test_params['theta'],
        v0=full_test_params['v0'],
        rate=full_test_params['rate'],
        div=full_test_params['div']
    )
    end_time = time.time()
    logging.info(f"FFT 'Actual' Price: {fft_price[0]:.6f} (calculated in {(end_time - start_time):.8f} seconds)")

    error = np.abs(chebyshev_price - fft_price[0])
    logging.info(f"Absolute Error: {error:.6f}")
