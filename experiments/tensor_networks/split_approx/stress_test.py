import logging
import json
import numpy as np

from bs_fourier_pricer import BSFourierPricer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


def stress_test():
    phi_tt_times = []
    vhat_tt_times = []
    online_tt_times = []
    fourier_times = []
    phi_samples = []
    vhat_samples = []
    tt_prices = []
    fourier_prices = []
    mean_tt_prices = []
    std_tt_prices = []
    rms_errors = []
    abs_errors = []

    dimensions = list(range(1, 10))
    etas = [0.5, 0.4, 0.4, 0.35, 0.3, 0.25, 0.3, 0.25, 0.2]
    base_seed = 42
    n_samples = 10
    s0_range = (90, 120)
    vol_range = (0.15, 0.25)

    phi_tt_params = {
        'eps': 1e-3,
        'rmax': 70,
        'max_iter': 100,
        'early_stopping_patience': 3,
        'early_stopping_tolerance': 1e-4,
        'return_info': True
    }
    vhat_tt_params = {
        'eps': 1e-3,
        'rmax': 50,
        'max_iter': 100,
        'early_stopping_patience': 3,
        'early_stopping_tolerance': 1e-4,
        'return_info': True
    }

    for i, (d, eta) in enumerate(zip(dimensions, etas)):

        run_tt_prices = []
        run_fourier_prices = []

        for j in range(n_samples):
            pricer = BSFourierPricer(
                d=dimensions[i],
                T=1.0,
                r=0.05,
                K=100,
                N=100,
                eta=etas[i],
                s0_range=s0_range,
                vol_range=vol_range,
                phi_tt_params=phi_tt_params,
                vhat_tt_params=vhat_tt_params,
                random_state=base_seed + j,
            )

            logging.info(f'Running stress test for dimension {d}, sample {j+1}/{n_samples}.')

            # TT gen
            phi_tt_info, phi_tt_runtime = pricer.run_phi_tt_cross()
            vhat_tt_info, vhat_tt_runtime = pricer.run_vhat_tt_cross()

            phi_tt_times.append(phi_tt_runtime)
            vhat_tt_times.append(vhat_tt_runtime)

            phi_samples.append(phi_tt_info['nsamples'])
            vhat_samples.append(vhat_tt_info['nsamples'])

            # Pricing
            tt_price, tt_runtime = pricer.price_from_tt()
            run_tt_prices.append(tt_price)

            if d < 5:
                f_price, f_runtime = pricer.price_from_fourier()
            else:
                f_runtime, f_price = None, None

            run_fourier_prices.append(f_price)

            online_tt_times.append(tt_runtime)
            fourier_times.append(f_runtime)

        tt_prices.append(run_tt_prices)
        fourier_prices.append(run_fourier_prices)

        mean_tt_prices.append(np.mean(run_tt_prices))
        std_tt_prices.append(np.std(run_tt_prices))

        if d < 5:
            errors = np.array(run_tt_prices) - np.array(run_fourier_prices)
            rms_errors.append(np.sqrt(np.mean(errors**2)))
            abs_errors.append(np.mean(np.abs(errors)))
        else:
            rms_errors.append(None)
            abs_errors.append(None)

    results = {
        'dimensions': dimensions,
        'etas': etas,
        'phi_tt_times': phi_tt_times,
        'vhat_tt_times': vhat_tt_times,
        'online_tt_times': online_tt_times,
        'fourier_times': fourier_times,
        'phi_samples': phi_samples,
        'vhat_samples': vhat_samples,
        'tt_prices': tt_prices,
        'fourier_prices': fourier_prices,
        'mean_tt_prices': mean_tt_prices,
        'std_tt_prices': std_tt_prices,
        'rms_errors': rms_errors,
        'abs_errors': abs_errors
    }

    with open('./output/stress_test_results3.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    stress_test()
