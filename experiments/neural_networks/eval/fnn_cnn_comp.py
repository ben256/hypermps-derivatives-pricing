import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_processing.dataset import TTDataset
from model.neural_mps import NeuralMPS, FNNNeuralMPS
from train.utils import setup_logging, find_dataset, create_recursive_folder


def eval_tt(tt_cores: List[torch.Tensor], x_indices: List[int]) -> torch.Tensor:
    """Contract a batch of TT cores at a chosen multi-index.

    Args:
        tt_cores: list length d, each tensor [batch, r_i, n, r_{i+1}]
        x_indices: list length d, each an int index into the physical dimension n

    Returns:
        Tensor [batch, 1, 1] (assuming boundary ranks are 1)
    """
    v = tt_cores[0][:, :, x_indices[0], :]  # [batch, r0, r1]
    for i in range(1, len(tt_cores)):
        v = v @ tt_cores[i][:, :, x_indices[i], :]  # successively [batch, r0, r_{i+1}]
    return v


def reconstruct_function(cores: List[torch.Tensor], N: int, d: int) -> torch.Tensor:
    """Reconstruct the (diagonal) function samples f(i,i,...,i) for i=0..N-1.

    Returns tensor [batch, N]."""
    batch = cores[0].shape[0]
    values = []
    for i in range(N):
        v = eval_tt(cores, [i] * d)  # [batch,1,1] if ranks boundaries are 1
        values.append(v.view(batch))
    return torch.stack(values, dim=1)  # [batch, N]


@dataclass
class EarlyStopping:
    patience: int = 5
    delta: float = 0.0
    offset: int = 0
    best_loss: Optional[float] = None
    counter: int = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stopped: bool = False

    def step(self, epoch: int, val_loss: float, model: nn.Module):
        if epoch < self.offset:
            return
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True


def build_tt_ranks(d: int, internal_rank: int) -> List[int]:
    """Return rank list including boundary 1's: [1, r, r, ..., r, 1]."""

    if d == 1:
        return [1, 1]
    return [1] + [internal_rank] * (d - 1) + [1]


def train_one_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        N: int,
        d: int,
        device: torch.device,
        epochs: int,
        lr: float,
        weight_decay: float,
        early_stopping: EarlyStopping,
        logger,
        tag: str,
) -> Tuple[List[float], List[float]]:
    criterion = nn.MSELoss()
    optimiser = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_hist, val_hist = [], []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for params, target in train_loader:
            params = params.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)  # [batch, N]

            optimiser.zero_grad()
            cores = model(params)
            output = reconstruct_function(cores, N, d)  # [batch, N]
            loss = criterion(output, target)
            loss.backward()
            optimiser.step()
            running += loss.item()

        avg_train = running / max(1, len(train_loader))
        train_hist.append(avg_train)

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for params, target in val_loader:
                params = params.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                cores = model(params)
                output = reconstruct_function(cores, N, d)
                val_running += criterion(output, target).item()
        avg_val = val_running / max(1, len(val_loader))
        val_hist.append(avg_val)

        logger.info(f'[{tag}] Epoch {epoch+1}/{epochs} Train {avg_train:.6f} Val {avg_val:.6f}')

        early_stopping.step(epoch, avg_val, model)
        if early_stopping.stopped:
            logger.info(f'[{tag}] Early stopping at epoch {epoch+1}')
            break

    # Load best weights if any
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    return train_hist, val_hist


def evaluate_model(
        model: nn.Module,
        data_loader: DataLoader,
        N: int,
        d: int,
        device: torch.device,
) -> Dict[str, float]:
    mse_sum = 0.0
    mae_sum = 0.0
    total = 0
    mse = nn.MSELoss(reduction='sum')
    mae = nn.L1Loss(reduction='sum')
    model.eval()
    with torch.no_grad():
        for params, target in data_loader:
            params = params.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)
            cores = model(params)
            output = reconstruct_function(cores, N, d)
            mse_sum += mse(output, target).item()
            mae_sum += mae(output, target).item()
            total += output.numel()
    mse_val = mse_sum / total
    mae_val = mae_sum / total
    rmse_val = np.sqrt(mse_val)
    return {'mse': mse_val, 'mae': mae_val, 'rmse': rmse_val}


def plot_sample_comparisons(
        cnn_model: nn.Module,
        fnn_model: nn.Module,
        test_dataset: TTDataset,
        N: int,
        d: int,
        device: torch.device,
        output_dir: str,
        num_samples: int = 4,
):
    indices = random.sample(range(len(test_dataset)), k=min(num_samples, len(test_dataset)))
    grid = np.linspace(-1, 1, N)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            params, target = test_dataset[idx]
            params = params.to(device, dtype=torch.float32).unsqueeze(0)
            target = target.to(device).cpu().numpy()

            cnn_pred = reconstruct_function(cnn_model(params), N, d).squeeze(0).cpu().numpy()
            fnn_pred = reconstruct_function(fnn_model(params), N, d).squeeze(0).cpu().numpy()

            ax.plot(grid, target, label='Target', color='black', linewidth=2)
            ax.plot(grid, cnn_pred, label='CNN', color='tab:blue', linestyle='--')
            ax.plot(grid, fnn_pred, label='FNN', color='tab:orange', linestyle='-.')
            ax.set_title(f'Sample {idx}')
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle('Target vs CNN & FNN Predictions', fontsize=16)
    plot_path = os.path.join(output_dir, 'cnn_fnn_comparison_samples.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description='Compare CNN-based NeuralMPS vs FNN-based NeuralMPS (semi-supervised TT).')
    parser.add_argument('--d', type=int, default=2, help='Dimensionality (start with 1 for 1D Gaussian).')
    parser.add_argument('--N', type=int, default=64, help='Number of grid points per dimension (TT physical size).')
    parser.add_argument('--max-rank', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cnn-decoder-type', type=str, choices=['shared', 'split'], default='shared')
    parser.add_argument('--dataset-dir', type=str, default='../data/datasets')
    parser.add_argument('--output-dir', type=str, default='../output')
    parser.add_argument('--correlation', type=float, default=0.3)
    parser.add_argument('--early-patience', type=int, default=20)
    parser.add_argument('--early-delta', type=float, default=0.0)
    parser.add_argument('--early-offset', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Logging & output
    comparison_dir = create_recursive_folder(args.output_dir, 'fnn_cnn_comp')
    logger = setup_logging(comparison_dir, 'comparison.log')
    logger.info('Starting FNN vs CNN NeuralMPS comparison (semi-supervised, TT)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}{f":{device.index}" if device.type == "cuda" else ""}')

    # Locate dataset (must be created beforehand with format=TT, semi_supervised=True)
    train_file, val_file, test_file, info = find_dataset(
        args.dataset_dir,
        d=args.d,
        N=args.N,
        correlation=args.correlation,
        format='TT',
        semi_supervised=True,
    )
    logger.info(f'Found dataset: train={train_file} val={val_file} test={test_file}')

    train_dataset = TTDataset(torch.load(train_file))
    val_dataset = TTDataset(torch.load(val_file))
    test_dataset = TTDataset(torch.load(test_file))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # TT ranks (ensure boundary 1's)
    ranks = build_tt_ranks(args.d, args.max_rank)
    logger.info(f'Ranks used: {ranks}')

    # Models
    cnn_model = NeuralMPS(
        ranks=ranks,
        n=args.N,  # physical dimension size
        input_size=info['input_size'],
        decoder_type=args.cnn_decoder_type,
        dropout=args.dropout,
    ).to(device)

    fnn_model = FNNNeuralMPS(
        ranks=ranks,
        n=args.N,
        input_size=info['input_size'],
        dropout=args.dropout,
    ).to(device)

    # Train CNN
    logger.info('Training CNN-based NeuralMPS...')
    cnn_es = EarlyStopping(patience=args.early_patience, delta=args.early_delta, offset=args.early_offset)
    cnn_train_hist, cnn_val_hist = train_one_model(
        cnn_model, train_loader, val_loader, args.N, args.d, device,
        args.epochs, args.learning_rate, args.weight_decay, cnn_es, logger, 'CNN'
    )

    # Train FNN
    logger.info('Training FNN-based NeuralMPS...')
    fnn_es = EarlyStopping(patience=args.early_patience, delta=args.early_delta, offset=args.early_offset)
    fnn_train_hist, fnn_val_hist = train_one_model(
        fnn_model, train_loader, val_loader, args.N, args.d, device,
        args.epochs, args.learning_rate, args.weight_decay, fnn_es, logger, 'FNN'
    )

    # Evaluate
    logger.info('Evaluating models on test set...')
    cnn_metrics = evaluate_model(cnn_model, test_loader, args.N, args.d, device)
    fnn_metrics = evaluate_model(fnn_model, test_loader, args.N, args.d, device)
    logger.info(f'CNN Metrics: {cnn_metrics}')
    logger.info(f'FNN Metrics: {fnn_metrics}')

    # Plot sample comparisons
    plot_path = plot_sample_comparisons(cnn_model, fnn_model, test_dataset, args.N, args.d, device, comparison_dir)
    logger.info(f'Saved sample comparison plot: {plot_path}')

    # Save artifacts
    with open(os.path.join(comparison_dir, 'metrics.json'), 'w') as f:
        json.dump({'cnn': cnn_metrics, 'fnn': fnn_metrics}, f, indent=2)
    with open(os.path.join(comparison_dir, 'loss_history.json'), 'w') as f:
        json.dump({
            'cnn': {'train': cnn_train_hist, 'val': cnn_val_hist},
            'fnn': {'train': fnn_train_hist, 'val': fnn_val_hist},
        }, f, indent=2)

    torch.save({'state_dict': cnn_model.state_dict(), 'ranks': ranks}, os.path.join(comparison_dir, 'cnn_model.pth'))
    torch.save({'state_dict': fnn_model.state_dict(), 'ranks': ranks}, os.path.join(comparison_dir, 'fnn_model.pth'))
    logger.info('Comparison complete.')


if __name__ == '__main__':
    main()

