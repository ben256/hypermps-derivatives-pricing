import argparse
import json
import math
import os
import time
import resource

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_processing.dataset import TTDataset
from eval.metrics import compute_metrics
from eval.plots import plot_parity, plot_residuals, plot_slices
from model.neural_mps import FNNNeuralMPS
from train.utils import (
    create_recursive_folder,
    setup_logging,
    find_dataset,
    EarlyStopping,
    eval_qtt,
    eval_tt, build_ranks,
)

import matplotlib.pyplot as plt


def count_model_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tt_storage_size(ranks: list[int], n: int) -> int:
    return int(sum(ranks[i] * n * ranks[i + 1] for i in range(len(ranks) - 1)))


def measure_latency_and_memory(model, dataloader, format: str, d: int, N: int, device: torch.device, warmup: int = 2, iters: int = 10):
    model.eval()
    timings = []

    # Select a single batch iterator reused
    data_iter = iter(dataloader)
    try:
        params, target = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        params, target = next(data_iter)

    params = params.to(device, dtype=torch.float32)
    target = target.to(device, dtype=torch.float32)

    def synchronize():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            tt_cores = model(params)
            if format == 'TT':
                _ = [eval_tt(tt_cores, [i] * d) for i in range(N)]
            else:
                _ = torch.stack([eval_qtt(tt_cores, [i] * d, N) for i in range(N)]).T

    # Baseline memory (ru_maxrss: bytes on macOS, kilobytes on Linux)
    import sys as _sys
    _rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    baseline_bytes = _rss if _sys.platform == 'darwin' else (_rss * 1024)

    with torch.no_grad():
        for _ in range(iters):
            synchronize()
            t0 = time.perf_counter()
            tt_cores = model(params)
            if format == 'TT':
                output = torch.stack([eval_tt(tt_cores, [i] * d) for i in range(N)], dim=1)
            else:
                output = torch.stack([eval_qtt(tt_cores, [i] * d, N) for i in range(N)]).T
            if isinstance(output, list):
                # Force materialization to avoid lazy timings
                _ = [o for o in output]
            else:
                _ = output.cpu()
            synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    # Peak memory during the run relative to baseline
    _rss2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = _rss2 if _sys.platform == 'darwin' else (_rss2 * 1024)
    delta_mb = max(0.0, (peak_bytes - baseline_bytes) / (1024.0 * 1024.0))  # MB

    avg_ms = float(np.mean(timings)) if timings else float('nan')
    std_ms = float(np.std(timings)) if timings else float('nan')
    return {
        'latency_ms_avg': avg_ms,
        'latency_ms_std': std_ms,
        'peak_memory_mb': delta_mb,
        'iters': iters,
        'batch_size': int(params.size(0)),
    }


def run_single_format(
    format: str,
    d: int,
    N: int,
    correlation: float,
    max_rank: int,
    batch_size: int,
    learning_rate: float,
    num_training_epochs: int,
    early_stopping_patience: int,
    early_stopping_delta: float,
    early_stopping_offset: int,
    weight_decay: float,
    dropout: float,
    dataset_dir: str,
    output_dir: str,
    seed: int,
    device: torch.device,
    logger,
):

    logger.info(f'Preparing dataset for format={format}')
    train_file, val_file, test_file, dataset_info = find_dataset(
        dataset_dir,
        d=d,
        N=N,
        correlation=correlation,
        format=format,
        semi_supervised=True,
        target_transform_type='zscore_log10',
    )

    logger.info('Loading datasets')
    train_dataset = TTDataset(torch.load(train_file))
    val_dataset = TTDataset(torch.load(val_file))
    test_dataset = TTDataset(torch.load(test_file))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    ranks, domain, n_model = build_ranks(
        format=format,
        d=d,
        N=N,
        max_rank=max_rank,
        device=device
    )

    logger.info(f'Ranks: {ranks[:5]}... len={len(ranks)} | n={n_model}')

    model = FNNNeuralMPS(
        ranks=ranks,
        n=n_model,
        input_size=dataset_info['input_size'],
        dropout=dropout,
        activation='tanh',
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loss_history, validation_loss_history = [], []
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        delta=early_stopping_delta,
        offset=early_stopping_offset,
    )

    logger.info(f'Starting training for {format}')
    for epoch in range(num_training_epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_idx, (params, target) in enumerate(train_dataloader):
            params = params.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            optimiser.zero_grad()
            tt_cores = model(params)
            if format == 'TT':
                output = [eval_tt(tt_cores, [i]*d) for i in range(N)]
            elif format == 'QTT':
                output = torch.stack([eval_qtt(tt_cores, [i]*d, N) for i in range(N)]).T

            loss = criterion(output, target)
            loss.backward()
            optimiser.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (params, target) in enumerate(val_dataloader):
                params = params.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)

                tt_cores = model(params)
                if format == 'TT':
                    output = [eval_tt(tt_cores, [i]*d) for i in range(N)]
                elif format == 'QTT':
                    output = torch.stack([eval_qtt(tt_cores, [i]*d, N) for i in range(N)]).T
                loss = criterion(output, target)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / max(1, len(val_dataloader))
        validation_loss_history.append(avg_val_loss)

        logger.info(f'[{format}] Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}')

        early_stopping.step(epoch, avg_val_loss, model)
        if early_stopping.stopped:
            logger.info(f'[{format}] Early stopping triggered at epoch {epoch}')
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    # Save artifacts
    final_model_path = os.path.join(output_dir, f'best_model_{format}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history,
    }, final_model_path)

    logger.info(f'[{format}] Training complete.')

    # Evaluate on test
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for params, target in test_dataloader:
            params = params.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            tt_cores = model(params)
            if format == 'TT':
                output = [eval_tt(tt_cores, [i]*d) for i in range(N)]
            elif format == 'QTT':
                output = torch.stack([eval_qtt(tt_cores, [i]*d, N) for i in range(N)]).T

            y_true_list.append(target.detach().cpu())
            y_pred_list.append(output.detach().cpu())

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    metrics, residuals, abs_err = compute_metrics(y_true, y_pred)

    # Plots
    plot_slices(y_true, y_pred, d, N, os.path.join(output_dir, f'slices_{format}'))
    plot_parity(y_true, y_pred, d, N, os.path.join(output_dir, f'parity_{format}'))
    plot_residuals(residuals, abs_err, metrics, d, N, os.path.join(output_dir, f'residuals_{format}'))

    # Latency and memory
    perf = measure_latency_and_memory(model, test_dataloader, format, d, N, device)

    # Size stats
    model_params = count_model_params(model)
    rep_size = tt_storage_size(ranks, n_model)
    num_cores = len(ranks) - 1

    # Save metrics
    results = {
        'format': format,
        'metrics': metrics,
        'model_params': model_params,
        'representation_size': rep_size,
        'num_cores': num_cores,
        'latency_ms_avg': perf['latency_ms_avg'],
        'latency_ms_std': perf['latency_ms_std'],
        'peak_memory_mb': perf['peak_memory_mb'],
        'd': d,
        'N': N,
        'ranks_len': len(ranks),
        'ranks_head': ranks[:5],
    }
    with open(os.path.join(output_dir, f'results_{format}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_accuracy_vs_model_size(tt_results, qtt_results, out_path):
    labels = ['TT', 'QTT']
    mse_vals = [tt_results['metrics']['MSE'], qtt_results['metrics']['MSE']]
    mae_vals = [tt_results['metrics']['MAE'], qtt_results['metrics']['MAE']]
    params_vals = [tt_results['model_params'], qtt_results['model_params']]
    cores_vals = [tt_results['num_cores'], qtt_results['num_cores']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    width = 0.35

    # Accuracy bars
    axes[0].bar(x - width/2, mse_vals, width, label='MSE')
    axes[0].bar(x + width/2, mae_vals, width, label='MAE')
    axes[0].set_title('Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Error')
    axes[0].legend()

    # Model size bars
    axes[1].bar(x - width/2, params_vals, width, label='#Params')
    axes[1].bar(x + width/2, cores_vals, width, label='#Cores')
    axes[1].set_title('Model Size')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Count')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def quantisation_testing(
    d: int,
    N: int,
    correlation: float,
    max_rank: int,
    batch_size: int,
    learning_rate: float,
    num_training_epochs: int,
    early_stopping_patience: int,
    early_stopping_delta: float,
    early_stopping_offset: int,
    weight_decay: float,
    dropout: float,
    dataset_dir: str,
    output_dir: str,
    seed: int,
):

    test_output = create_recursive_folder(output_dir, 'quantisation_testing')
    logger = setup_logging(test_output, 'quantisation.log')
    logger.info('Starting Quantized (QTT) vs Non-Quantized (TT) benchmark')

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}{f":{device.index}" if device.type == "cuda" else ""}')

    # Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f'Set seed to {seed}')

    # Ensure N is a power of two for QTT
    if int(math.log2(N)) != math.log2(N):
        raise ValueError('N must be a power of 2 for QTT experiments.')

    # Run TT and QTT
    qtt_results = run_single_format(
        format='QTT',
        d=d,
        N=N,
        correlation=correlation,
        max_rank=max_rank,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_training_epochs=num_training_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
        early_stopping_offset=early_stopping_offset,
        weight_decay=weight_decay,
        dropout=dropout,
        dataset_dir=dataset_dir,
        output_dir=test_output,
        seed=seed,
        device=device,
        logger=logger,
    )

    tt_results = run_single_format(
        format='TT',
        d=d,
        N=N,
        correlation=correlation,
        max_rank=max_rank,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_training_epochs=num_training_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_delta=early_stopping_delta,
        early_stopping_offset=early_stopping_offset,
        weight_decay=weight_decay,
        dropout=dropout,
        dataset_dir=dataset_dir,
        output_dir=test_output,
        seed=seed,
        device=device,
        logger=logger,
    )

    # Compression ratio (TT storage vs QTT storage)
    comp_ratio = tt_results['representation_size'] / max(1, qtt_results['representation_size'])

    summary = {
        'TT': tt_results,
        'QTT': qtt_results,
        'compression_ratio_tt_over_qtt': comp_ratio,
    }

    with open(os.path.join(test_output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save a small CSV table for compression
    with open(os.path.join(test_output, 'compression_table.csv'), 'w') as f:
        f.write('format,representation_size,num_cores,model_params,latency_ms_avg,peak_memory_mb\n')
        f.write(f"TT,{tt_results['representation_size']},{tt_results['num_cores']},{tt_results['model_params']},{tt_results['latency_ms_avg']:.3f},{tt_results['peak_memory_mb']:.3f}\n")
        f.write(f"QTT,{qtt_results['representation_size']},{qtt_results['num_cores']},{qtt_results['model_params']},{qtt_results['latency_ms_avg']:.3f},{qtt_results['peak_memory_mb']:.3f}\n")

    # Plot accuracy vs model size
    plot_accuracy_vs_model_size(tt_results, qtt_results, os.path.join(test_output, 'accuracy_vs_model_size.png'))

    logger.info('Benchmark complete. Outputs written to: %s', test_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--max-rank', type=int, default=10)
    parser.add_argument('--correlation', type=float, default=0.5)
    parser.add_argument('--dataset-dir', type=str, default='../data/datasets')
    parser.add_argument('--output-dir', type=str, default='../output')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--num-training-epochs', type=int, default=150)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--early-stopping-delta', type=float, default=1e-5)
    parser.add_argument('--early-stopping-offset', type=int, default=40)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    quantisation_testing(
        d=args.d,
        N=args.N,
        correlation=args.correlation,
        max_rank=args.max_rank,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_training_epochs=args.num_training_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        early_stopping_offset=args.early_stopping_offset,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()

