import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_processing.dataset import TTDataset
from model.neural_mps import NeuralMPS
from train.utils import setup_logging, find_dataset


def eval_tt(tt_cores, x_indices):
    """Evaluates a TT at given indices."""
    v = tt_cores[0][:, :, x_indices[0], :]
    for i in range(1, len(tt_cores)):
        v = v @ tt_cores[i][:, :, x_indices[i], :]
    return v.squeeze()


def eval_btt(btt_cores, x_indices, N):
    """Evaluates a BTT at given indices."""
    k = int(np.log2(N))
    binary_indices = []
    for idx in x_indices:
        binary_indices.extend([int(b) for b in bin(idx)[2:].zfill(k)])

    v = btt_cores[0][:, :, binary_indices[0], :]
    for i in range(1, len(btt_cores)):
        v = v @ btt_cores[i][:, :, binary_indices[i], :]
    return v.squeeze()


def test(
        d: int = 4,
        N: int = 64,
        correlation: float = 0.3,
        max_rank: int = 20,
        format: str = 'BTT',

        batch_size:int = 200,
        decoder_type: str = 'split',
        dataset_dir: str = '../data/datasets',
        output_dir: str = '../output',
        model_dir: str = '../data/models/training_2'
):
    logger = setup_logging(output_dir)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}{f":{device.index}" if device.type == "cuda" else ""}')

    _, _, test_file, dataset_info = find_dataset(
        dataset_dir,
        d=d,
        N=N,
        correlation=correlation,
        format=format,
        semi_supervised=True,
    )

    test_dataset = TTDataset(torch.load(test_file))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if format == 'TT':
        domain = [torch.arange(N, device=device) for _ in range(d)]
        ranks = [max_rank] * (d - 1)
        n_model = N
    elif format == 'BTT':
        k = int(np.log2(N))
        n_model = 2
        domain = [torch.arange(2, device=device) for _ in range(d * k)]
        ranks = [1]
        for i in range(d * k):
            if len(ranks) < (d * k) // 2:
                ranks.append(min(ranks[-1]*2, max_rank))
            else:
                ranks.append(min(ranks[-1]*2, max_rank))
                break
        ranks.extend(ranks[::-1][1:])

    else:
        raise ValueError(f"Unsupported format: {format}")

    model_data = torch.load(f'{model_dir}/best_model.pth', map_location=device)
    model = NeuralMPS(
        ranks=ranks,
        n=n_model,
        input_size=dataset_info['input_size'],
        decoder_type=decoder_type,
    )
    model.to(device)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    sum_squared_error = 0.0
    sum_absolute_error = 0.0
    total_elements = 0
    mse_loss_fn = nn.MSELoss(reduction='sum')
    mae_loss_fn = nn.L1Loss(reduction='sum')

    model.eval()
    with torch.no_grad():
        for batch_idx, (params, target) in enumerate(test_dataloader):
            params = params.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            tt_cores = model(params)
            if format == 'TT':
                output = torch.stack([eval_tt(tt_cores, [i]*d) for i in range(N)]).squeeze().T
            elif format == 'BTT':
                output = torch.stack([eval_btt(tt_cores, [i]*d, N) for i in range(N)]).squeeze().T

            # accumulate batch errors
            sum_squared_error += mse_loss_fn(output, target).item()
            sum_absolute_error += mae_loss_fn(output, target).item()
            total_elements += output.numel()

    # compute final metrics
    test_mse = sum_squared_error / total_elements
    test_mae = sum_absolute_error / total_elements
    logger.info(f'Test MSE: {test_mse:.8f}')
    logger.info(f'Test MAE: {test_mae:.8f}')

    logger.info('Generating plots for 8 random samples...')
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    axes = axes.flatten()

    # Get 8 random samples
    num_samples_to_plot = 8
    random_indices = np.random.choice(len(test_dataset), num_samples_to_plot, replace=False)
    random_samples = [test_dataset[i] for i in random_indices]

    grid_1d = np.linspace(-1, 1, N)

    with torch.no_grad():
        for i, (params, target) in enumerate(random_samples):
            ax = axes[i]
            params = params.to(device, dtype=torch.float32).unsqueeze(0) # Add batch dimension
            target = target.to(device, dtype=torch.float32)

            tt_cores = model(params)
            if format == 'TT':
                output = torch.stack([eval_tt(tt_cores, [j] * d) for j in range(N)]).squeeze()
            elif format == 'BTT':
                output = torch.stack([eval_btt(tt_cores, [j] * d, N) for j in range(N)]).squeeze()

            ax.plot(grid_1d, target.cpu().numpy(), label='Target Function', color='blue', linewidth=2)
            ax.plot(grid_1d, output.cpu().numpy(), label='NN Output', color='orange', linestyle='--', linewidth=2)
            ax.set_title(f'Sample {i + 1}')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.grid(True)
            ax.legend()

    fig.tight_layout()
    plt.suptitle('Target Function vs. NN Output', fontsize=16, y=1.02)
    plot_path = os.path.join(output_dir, 'test_plots.png')
    plt.savefig(plot_path)
    logger.info(f'Saved plots to {plot_path}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--N', type=int, default=64)
    parser.add_argument('--correlation', type=float, default=0.3)
    parser.add_argument('--max-rank', type=int, default=20)
    parser.add_argument('--format', type=str, choices=['TT', 'BTT'], default='BTT')

    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--decoder-type', type=str, choices=['shared', 'split'], default='split')
    parser.add_argument('--dataset-dir', type=str, default='../data/datasets')
    parser.add_argument('--output-dir', type=str, default='../output')
    parser.add_argument('--model-dir', type=str, default='../data/models/training_3')

    args = parser.parse_args()

    test(
        d=args.d,
        N=args.N,
        correlation=args.correlation,
        max_rank=args.max_rank,
        format=args.format,
        batch_size=args.batch_size,
        decoder_type=args.decoder_type,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
    )
