import json
import logging
import math
from typing import Dict, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats

from functions import sample_gaussian_params_batch, gaussian_analytical
from utils import idx_to_bits


def plot_slices(model, grid, d, N, device, output_dir, step, rng, num_samples=4):
    model.eval()
    with torch.no_grad():
        L = int(math.log2(N))
        center = N // 2

        conditioning_params, params_list = sample_gaussian_params_batch(num_samples, rng, d)
        params_plot = conditioning_params.to(device, dtype=torch.float32)

        fig = plt.figure(figsize=(4 * num_samples, 3 * d))
        x_axis = grid.detach().cpu().numpy()

        for c in range(num_samples):
            for dim in range(d):
                coords = []
                for j in range(d):
                    if j == dim:
                        coords.append(torch.arange(N, device=device, dtype=torch.long))
                    else:
                        coords.append(torch.full((N,), center, device=device, dtype=torch.long))

                idx_nd_slice = torch.stack(coords, dim=-1).unsqueeze(0)  # [1, N, d]
                bits_slice = idx_to_bits(idx_nd_slice, d=d, L=L)  # [1, N, K]

                pred_slice = model(params_plot[c:c+1], bits_slice).squeeze(0).detach().cpu().numpy()

                y_true_slice = gaussian_analytical(
                    idx_nd_slice, grid, [params_list[c]]
                ).squeeze(0).detach().cpu().numpy()

                ax = fig.add_subplot(d, num_samples, dim * num_samples + c + 1)
                ax.plot(x_axis, y_true_slice, label='target', lw=2)
                ax.plot(x_axis, pred_slice, label='pred', lw=2)

                if dim == d - 1:
                    ax.set_xlabel('x')
                if c == 0:
                    ax.set_ylabel(f'log-density (dim {dim})')
                if dim == 0:
                    ax.set_title(f'sample {c}')

                if dim == 0 and c == num_samples - 1:
                    ax.legend()

        fig.tight_layout()
        out_path = f'{output_dir}/slices_step_{step}.png'
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    model.train()


def compute_evaluation_metrics(
        model,
        grid: torch.Tensor,
        d: int,
        N: int,
        device: torch.device,
        rng: torch.Generator = None,
        n_test_samples: int = 1000,
        n_test_distributions: int = 100,
        test_bits: torch.Tensor = None,
        test_params_list: list = None,
        test_conditioning_params: torch.Tensor = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for the trained model.

    Args:
        model: Trained model (HyperHyperNetwork or BenchmarkMPS)
        grid: Grid tensor [N]
        d: Dimensionality
        N: Grid size
        device: Device to run on
        rng: Random generator (used only if test data not provided)
        n_test_samples: Number of sample points per distribution
        n_test_distributions: Number of different distributions to test
        test_bits: Optional pre-generated test bits [n_test_distributions, n_test_samples, K]
        test_params_list: Optional pre-generated parameter list
        test_conditioning_params: Optional pre-generated conditioning params [n_test_distributions, conditioning_dim]

    Returns:
        Dictionary containing all metrics
    """
    from functions import sample_gaussian_params_batch, gaussian_analytical
    from utils import bits_to_idx

    model.eval()

    L = int(math.log2(N))
    K = d * L

    # Determine if we're using pre-generated data or generating on the fly
    using_fixed_data = (test_bits is not None and
                        test_params_list is not None and
                        test_conditioning_params is not None)

    if not using_fixed_data and rng is None:
        raise ValueError("Either provide test data or provide an rng for random generation")

    metrics = {
        'mse': [],
        'mae': [],
        'mape': [],
        'max_abs_error': [],
        'r2_score': [],
        'pearson_corr': [],
        'log_likelihood_ratio': [],
        'relative_l2_error': [],
    }

    with torch.no_grad():
        for i in range(n_test_distributions):
            if using_fixed_data:
                # Use pre-generated test data
                bits = test_bits[i:i+1]  # [1, n_test_samples, K]
                indices = bits_to_idx(bits, d, L)
                conditioning_params = test_conditioning_params[i:i+1]  # [1, conditioning_dim]
                params_list = [test_params_list[i]]
            else:
                # Generate random data on the fly (old behavior)
                bits = torch.randint(0, 2, (1, n_test_samples, K), device=device, dtype=torch.long)
                indices = bits_to_idx(bits, d, L)
                conditioning_params, params_list = sample_gaussian_params_batch(1, rng, d)
                conditioning_params = conditioning_params.to(device, dtype=torch.float32)

            # Get predictions
            predictions = model.forward(conditioning_params, bits).squeeze(0)  # [n_test_samples]

            # Get ground truth
            targets = gaussian_analytical(indices, grid, params_list).squeeze(0)  # [n_test_samples]

            # Move to CPU for metric computation
            pred_cpu = predictions.cpu().numpy()
            target_cpu = targets.cpu().numpy()

            # MSE (Mean Squared Error)
            mse = np.mean((pred_cpu - target_cpu) ** 2)
            metrics['mse'].append(mse)

            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(pred_cpu - target_cpu))
            metrics['mae'].append(mae)

            # MAPE (Mean Absolute Percentage Error) - with safety for near-zero values
            epsilon = 1e-8
            mape = np.mean(np.abs((pred_cpu - target_cpu) / (np.abs(target_cpu) + epsilon))) * 100
            metrics['mape'].append(mape)

            # Max Absolute Error
            max_abs_err = np.max(np.abs(pred_cpu - target_cpu))
            metrics['max_abs_error'].append(max_abs_err)

            # R² Score (Coefficient of Determination)
            ss_res = np.sum((target_cpu - pred_cpu) ** 2)
            ss_tot = np.sum((target_cpu - np.mean(target_cpu)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + epsilon))
            metrics['r2_score'].append(r2)

            # Pearson Correlation Coefficient
            if len(pred_cpu) > 1:
                pearson_corr, _ = stats.pearsonr(pred_cpu, target_cpu)
                metrics['pearson_corr'].append(pearson_corr)

            # Log-Likelihood Ratio (for probabilistic models)
            # This measures how well the model approximates the true log-density
            ll_ratio = np.mean(np.abs(pred_cpu - target_cpu) / (np.abs(target_cpu) + epsilon))
            metrics['log_likelihood_ratio'].append(ll_ratio)

            # Relative L2 Error
            l2_error = np.linalg.norm(pred_cpu - target_cpu) / (np.linalg.norm(target_cpu) + epsilon)
            metrics['relative_l2_error'].append(l2_error)

    # Aggregate metrics across all test distributions
    aggregated_metrics = {}
    for key, values in metrics.items():
        aggregated_metrics[f'{key}_mean'] = float(np.mean(values))
        aggregated_metrics[f'{key}_std'] = float(np.std(values))
        aggregated_metrics[f'{key}_median'] = float(np.median(values))
        aggregated_metrics[f'{key}_min'] = float(np.min(values))
        aggregated_metrics[f'{key}_max'] = float(np.max(values))

    model.train()
    return aggregated_metrics


def compute_grid_coverage_metrics(
        model,
        grid: torch.Tensor,
        d: int,
        N: int,
        device: torch.device,
        rng: torch.Generator,
        n_distributions: int = 10,
) -> Dict[str, float]:
    """
    Compute metrics on full grid coverage to assess overall quality.

    Args:
        model: Trained model
        grid: Grid tensor
        d: Dimensionality
        N: Grid size
        device: Device
        rng: Random generator
        n_distributions: Number of distributions to test

    Returns:
        Dictionary of grid coverage metrics
    """
    from functions import sample_gaussian_params_batch, gaussian_analytical
    from utils import idx_to_bits

    model.eval()

    L = int(math.log2(N))

    metrics = {
        'grid_mse': [],
        'grid_mae': [],
        'grid_max_error': [],
    }

    with torch.no_grad():
        for _ in range(n_distributions):
            # Create full grid indices for 1D slices through center
            center = N // 2

            all_errors = []

            for dim in range(d):
                # Create indices for a 1D slice along dimension 'dim'
                coords = []
                for j in range(d):
                    if j == dim:
                        coords.append(torch.arange(N, device=device, dtype=torch.long))
                    else:
                        coords.append(torch.full((N,), center, device=device, dtype=torch.long))

                idx_nd_slice = torch.stack(coords, dim=-1).unsqueeze(0)  # [1, N, d]
                bits_slice = idx_to_bits(idx_nd_slice, d=d, L=L)  # [1, N, K]

                # Generate parameters
                conditioning_params, params_list = sample_gaussian_params_batch(1, rng, d)
                conditioning_params = conditioning_params.to(device, dtype=torch.float32)

                # Predictions
                pred_slice = model(conditioning_params, bits_slice).squeeze(0)  # [N]

                # Ground truth
                target_slice = gaussian_analytical(idx_nd_slice, grid, params_list).squeeze(0)  # [N]

                # Compute errors
                errors = torch.abs(pred_slice - target_slice).cpu().numpy()
                all_errors.extend(errors)

                # Slice-specific metrics
                mse = torch.mean((pred_slice - target_slice) ** 2).item()
                mae = torch.mean(torch.abs(pred_slice - target_slice)).item()
                max_err = torch.max(torch.abs(pred_slice - target_slice)).item()

                metrics['grid_mse'].append(mse)
                metrics['grid_mae'].append(mae)
                metrics['grid_max_error'].append(max_err)

    # Aggregate
    aggregated = {}
    for key, values in metrics.items():
        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
        aggregated[f'{key}_max'] = float(np.max(values))

    model.train()
    return aggregated


def save_metrics(metrics: Dict[str, float], output_path: str):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f'Metrics saved to {output_path}')


def load_metrics(metrics_path: str) -> Dict[str, float]:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def print_metrics_summary(metrics: Dict[str, float], title: str = 'Evaluation Metrics'):
    """Print a formatted summary of metrics."""
    logging.info('')
    logging.info(f'{"="*60}')
    logging.info(f'{title:^60}')
    logging.info(f'{"="*60}')

    # Group metrics by type
    metric_groups = {
        'MSE': [],
        'MAE': [],
        'MAPE': [],
        'Max Abs Error': [],
        'R² Score': [],
        'Pearson Correlation': [],
        'Log-Likelihood Ratio': [],
        'Relative L2 Error': [],
        'Grid Coverage': [],
    }

    for key, value in sorted(metrics.items()):
        if 'mse' in key:
            metric_groups['MSE'].append((key, value))
        elif 'mae' in key and 'grid' not in key:
            metric_groups['MAE'].append((key, value))
        elif 'mape' in key:
            metric_groups['MAPE'].append((key, value))
        elif 'max' in key and 'grid' not in key:
            metric_groups['Max Abs Error'].append((key, value))
        elif 'r2' in key:
            metric_groups['R² Score'].append((key, value))
        elif 'pearson' in key:
            metric_groups['Pearson Correlation'].append((key, value))
        elif 'log_likelihood' in key:
            metric_groups['Log-Likelihood Ratio'].append((key, value))
        elif 'relative_l2' in key:
            metric_groups['Relative L2 Error'].append((key, value))
        elif 'grid' in key:
            metric_groups['Grid Coverage'].append((key, value))

    for group_name, group_metrics in metric_groups.items():
        if group_metrics:
            logging.info('')
            logging.info(f'{group_name}:')
            logging.info(f'{"-"*60}')
            for metric_name, metric_value in group_metrics:
                logging.info(f'  {metric_name:45s}: {metric_value:12.6f}')

    logging.info('')
    logging.info(f'{"="*60}')
    logging.info('')


def compare_models(
        metrics_dict: Dict[str, Dict[str, float]],
        primary_metrics: Optional[list] = None
):
    """
    Compare multiple models based on their metrics.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        primary_metrics: List of metric names to focus on for comparison
    """
    if primary_metrics is None:
        primary_metrics = [
            'mse_mean', 'mae_mean', 'r2_score_mean',
            'pearson_corr_mean', 'relative_l2_error_mean'
        ]

    logging.info(f'\n{"="*80}')
    logging.info(f'{"Model Comparison":^80}')
    logging.info(f'{"="*80}\n')

    # Create comparison table
    model_names = list(metrics_dict.keys())

    for metric in primary_metrics:
        logging.info(f'\n{metric}:')
        logging.info(f'{"-"*80}')

        values = []
        for model_name in model_names:
            if metric in metrics_dict[model_name]:
                value = metrics_dict[model_name][metric]
                values.append((model_name, value))

        # Sort by value (lower is better for most metrics except R² and correlation)
        reverse = 'r2_score' in metric or 'pearson_corr' in metric
        values.sort(key=lambda x: x[1], reverse=reverse)

        for rank, (model_name, value) in enumerate(values, 1):
            marker = '🏆' if rank == 1 else '  '
            logging.info(f'  {marker} {rank}. {model_name:30s}: {value:12.6f}')

    logging.info(f'\n{"="*80}\n')
