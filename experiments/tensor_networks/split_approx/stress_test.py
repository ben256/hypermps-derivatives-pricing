import logging
import json
import numpy as np

from bs_fourier_pricer import BSFourierPricer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def stress_test():
    dimensions = list(range(1, 10))
    etas = [0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]
    base_seed = 42
    n_samples = 10
    s0_range = (90, 120)
    vol_range = (0.15, 0.25)

    phi_tt_params = {'eps': 1e-3, 'rmax': 50, 'max_iter': 100, 'early_stopping_patience': 3, 'early_stopping_tolerance': 1e-4, 'return_info': True}
    vhat_tt_params = {'eps': 1e-3, 'rmax': 50, 'max_iter': 100, 'early_stopping_patience': 3, 'early_stopping_tolerance': 1e-4, 'return_info': True}

    results = {
        'dimensions': dimensions,
        'etas': etas[:len(dimensions)],
        'phi_tt_times': [], 'vhat_tt_times': [], 'online_tt_times': [],
        'fourier_times': [], 'phi_samples': [], 'vhat_samples': [],
        'tt_prices': [], 'fourier_prices': [], 'mean_tt_prices': [],
        'std_tt_prices': [], 'rms_errors': [], 'abs_errors': []
    }

    for i, (d, eta) in enumerate(zip(dimensions, etas)):
        run_phi_tt_times, run_vhat_tt_times, run_online_tt_times = [], [], []
        run_fourier_times, run_phi_samples, run_vhat_samples = [], [], []
        run_tt_prices, run_fourier_prices = [], []

        for j in range(n_samples):
            logging.info(f'Running stress test for dimension {d}, sample {j+1}/{n_samples}.')
            pricer = BSFourierPricer(
                d=d, T=1.0, r=0.05, K=100, N=100, eta=eta,
                s0_range=s0_range, vol_range=vol_range,
                phi_tt_params=phi_tt_params, vhat_tt_params=vhat_tt_params,
                random_state=base_seed + j,
            )

            phi_tt_info, phi_tt_runtime = pricer.run_phi_tt_cross()
            vhat_tt_info, vhat_tt_runtime = pricer.run_vhat_tt_cross()
            tt_price, tt_runtime = pricer.price_from_tt()

            run_phi_tt_times.append(phi_tt_runtime)
            run_vhat_tt_times.append(vhat_tt_runtime)
            run_online_tt_times.append(tt_runtime)
            run_phi_samples.append(phi_tt_info['nsamples'])
            run_vhat_samples.append(vhat_tt_info['nsamples'])
            run_tt_prices.append(tt_price)

            if d < 5:
                f_price, f_runtime = pricer.price_from_fourier()
                run_fourier_prices.append(f_price)
                run_fourier_times.append(f_runtime)
            else:
                run_fourier_prices.append(None)
                run_fourier_times.append(None)

        results['phi_tt_times'].append(run_phi_tt_times)
        results['vhat_tt_times'].append(run_vhat_tt_times)
        results['online_tt_times'].append(run_online_tt_times)
        results['fourier_times'].append(run_fourier_times)
        results['phi_samples'].append(run_phi_samples)
        results['vhat_samples'].append(run_vhat_samples)
        results['tt_prices'].append(run_tt_prices)
        results['fourier_prices'].append(run_fourier_prices)

        results['mean_tt_prices'].append(np.mean(run_tt_prices))
        results['std_tt_prices'].append(np.std(run_tt_prices))

        if d < 5:
            errors = np.array(run_tt_prices) - np.array(run_fourier_prices)
            results['rms_errors'].append(np.sqrt(np.mean(errors**2)))
            results['abs_errors'].append(np.max(np.abs(errors)))
        else:
            results['rms_errors'].append(None)
            results['abs_errors'].append(None)

    with open('./output/stress_test_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    stress_test()