import argparse
import json
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_processing.dataset import TTDataset
from eval.metrics import compute_metrics
from eval.plots import plot_parity, plot_residuals, plot_slices
from model.neural_mps import FNNNeuralMPS, NeuralMPS
from train.utils import create_recursive_folder, setup_logging, find_dataset, EarlyStopping, eval_qtt, eval_tt


def compare_activations(
        d: int,
        N: int,
        correlation: float,
        max_rank: int,
        format: str,
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
        seed: int
):

    test_output = create_recursive_folder(output_dir, 'compare_activation')
    logger = setup_logging(test_output, 'benchmark.log')
    logger.info('Starting activation function comparison benchmark')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f'Using device: {device}{f":{device.index}" if device.type == "cuda" else ""}')

    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f'Set seed to {seed}')

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

    logger.info('Setting up model parameters')
    if format == 'TT':
        domain = [torch.arange(N, device=device) for _ in range(d)]
        ranks = [max_rank] * (d - 1)
        n_model = N
    elif format == 'QTT':
        k = int(np.log2(N))
        n_model = 2
        domain = [torch.arange(2, device=device) for _ in range(d * k)]
        ranks = [1]
        for i in range(d * k):
            if len(ranks) < (d * k) // 2:
                ranks.append(min(ranks[-1] * 2, max_rank))
            else:
                ranks.append(min(ranks[-1] * 2, max_rank))
                break
        ranks.extend(ranks[::-1][1:])
    else:
        raise ValueError(f"Unsupported format: {format}")

    activations = ['relu', 'tanh', 'linear']

    for activation in activations:
        logger.info(f'Starting training with activation: {activation}')
        model = FNNNeuralMPS(
            ranks=ranks,
            n=n_model,
            input_size=dataset_info['input_size'],
            dropout=dropout,
            activation=activation,
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

        logger.info(f'Activation: {activation}')
        logger.info(f'Number of training epochs: {num_training_epochs}')
        logger.info(f'Batch size: {batch_size}')
        logger.info(f'Learning rate: {learning_rate}')
        logger.info(f'Early stopping: Patience: {early_stopping_patience}, Delta: {early_stopping_delta}, Offset: {early_stopping_offset}')
        logger.info(f'Weight decay: {weight_decay}')
        logger.info(f'Dropout: {dropout}')
        logger.info(f'Dataset directory: {dataset_dir}')
        logger.info(f'Output directory: {test_output}')

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

            avg_train_loss = epoch_train_loss / len(train_dataloader)
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

            avg_val_loss = epoch_val_loss / len(val_dataloader)
            validation_loss_history.append(avg_val_loss)

            logger.info(f'Epoch: {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}')

            with open(f'{test_output}/loss.json', 'w') as f:
                json.dump({'train': train_loss_history, 'validation': validation_loss_history}, f)

            early_stopping.step(epoch, avg_val_loss, model)
            if early_stopping.stopped:
                logger.info(f'Early stopping triggered at epoch {epoch}')
                break

        if early_stopping.best_state is not None:
            model.load_state_dict(early_stopping.best_state)

        final_model_path = os.path.join(test_output, 'best_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'train_loss_history': train_loss_history,
            'validation_loss_history': validation_loss_history,
        }, final_model_path)

        logger.info('Training complete.')

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
        metrics_path = os.path.join(test_output, f'metrics_{activation}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        plot_slices(y_true, y_pred, d, N, os.path.join(test_output, f'slices_{activation}'))
        plot_parity(y_true, y_pred, d, N,  os.path.join(test_output, f'parity_{activation}'))
        plot_residuals(residuals, abs_err, metrics, d, N, os.path.join(test_output, f'residuals_{activation}'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--max-rank', type=int, default=10)
    parser.add_argument('--correlation', type=float, default=0.5)
    parser.add_argument('--format', type=str, choices=['TT', 'QTT'], default='QTT')
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

    compare_activations(
        d=args.d,
        N=args.N,
        correlation=args.correlation,
        max_rank=args.max_rank,
        format=args.format,
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
        seed=args.seed
    )


if __name__ == '__main__':
    main()