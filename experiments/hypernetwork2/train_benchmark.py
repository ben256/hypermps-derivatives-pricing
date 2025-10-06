import argparse
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from functions import sample_gaussian_params_batch, gaussian_analytical
from benchmark import BenchmarkMPS
from utils import (
    seed_everything,
    create_recursive_folder,
    setup_logging,
    setup_device,
    gaussian_conditioning_param_len,
    bits_to_idx
)
from eval import (
    compute_evaluation_metrics,
    save_metrics,
    print_metrics_summary,
    plot_slices
)


def train(
        N: int,
        d: int,
        grid_min: float,
        grid_max: float,
        r: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        decoder_type: str,
        batch_size: int,
        n_training_steps: int,
        n_warmup_steps: int,
        learning_rate: float,
        n_train_samples: int,
        output_dir: str,
        seed: int,
):
    seed_everything(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    output_dir = create_recursive_folder(output_dir, 'benchmark')
    logger = setup_logging(output_dir)
    device = setup_device()

    logger.info('Initialising benchmark model training')
    logger.info(f'Using device: {device}')
    logger.info(f'Output directory: {output_dir}')

    conditioning_dim = gaussian_conditioning_param_len(d)

    L = int(math.log2(N))
    K = d * L
    grid = torch.linspace(grid_min, grid_max, N, device=device, dtype=torch.float32)

    model = BenchmarkMPS(
        N=N,
        r=r,
        d=d,
        conditioning_dim=conditioning_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        decoder_type=decoder_type,
    ).to(device=device, dtype=torch.float32)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    warmup = LambdaLR(optimizer, lambda step: min((step + 1) / max(1, n_warmup_steps), 1.0))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_training_steps - n_warmup_steps), eta_min=3e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[n_warmup_steps])

    logger.info(f'Training parameters:')
    logger.info(f'  N: {N}')
    logger.info(f'  d: {d}')
    logger.info(f'  r: {r}')
    logger.info(f'  hidden_size: {hidden_size}')
    logger.info(f'  num_layers: {num_layers}')
    logger.info(f'  dropout: {dropout}')
    logger.info(f'  decoder_type: {decoder_type}')
    logger.info(f'  batch_size: {batch_size}')
    logger.info(f'  n_training_steps: {n_training_steps}')
    logger.info(f'  n_warmup_steps: {n_warmup_steps}')
    logger.info(f'  learning_rate: {learning_rate}')
    logger.info(f'  n_train_samples: {n_train_samples}')

    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    logger.info('Starting model training')

    step = 0
    model.train()

    rolling_window = 250
    loss_buffer = deque(maxlen=rolling_window)
    loss_sum = 0.0

    while step < n_training_steps:

        bits = torch.randint(0, 2, (batch_size, n_train_samples, K), device=device, dtype=torch.long)
        indices = bits_to_idx(bits, d, L)

        conditioning_params, params_list = sample_gaussian_params_batch(batch_size, rng, d)
        conditioning_params = conditioning_params.to(device, dtype=torch.float32)
        predictions = model.forward(conditioning_params, bits)  # [batch_size, S]

        targets = gaussian_analytical(indices, grid, params_list)

        loss = F.smooth_l1_loss(predictions, targets, beta=0.5, reduction='mean')

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_clip):
            logger.warning('Non-finite grad_norm detected, skipping update this step.')
            optimizer.zero_grad(set_to_none=True)
            step += 1
            continue

        optimizer.step()
        scheduler.step()

        step += 1

        loss_value = loss.item()
        if len(loss_buffer) == rolling_window:
            loss_sum -= loss_buffer[0]
        loss_buffer.append(loss_value)
        loss_sum += loss_value

        if step % 100 == 0:
            avg_loss = loss_sum / len(loss_buffer)
            lr = optimizer.param_groups[0]['lr']
            logger.info(f'Step {step}/{n_training_steps} | Loss: {avg_loss:.6f} | LR: {lr:.2e}')

        if step % 2000 == 0 or step == n_training_steps:
            logger.info(f'Plotting slices at step {step}')
            plot_slices(model, grid, d, N, device, output_dir, step, rng, num_samples=4)

    logger.info('Training completed')

    logger.info('Computing final evaluation metrics...')
    metrics = compute_evaluation_metrics(
        model=model,
        grid=grid,
        d=d,
        N=N,
        device=device,
        rng=rng,
        n_test_samples=1000,
        n_test_distributions=100,
    )

    save_metrics(metrics, f'{output_dir}/evaluation_metrics.json')
    print_metrics_summary(metrics, 'Final Model Evaluation')

    model_path = f'{output_dir}/model_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'N': N,
            'r': r,
            'd': d,
            'conditioning_dim': conditioning_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
        },
        'metrics': metrics,
    }, model_path)
    logger.info(f'Model saved to {model_path}')

    logger.info('Benchmark training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Grid and problem parameters
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--grid_min', type=float, default=-4.0)
    parser.add_argument('--grid_max', type=float, default=4.0)

    # Model architecture
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--decoder_type', type=str, default='shared', choices=['core', 'dimension', 'shared'])

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_training_steps', type=int, default=20000)
    parser.add_argument('--n_warmup_steps', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_train_samples', type=int, default=1024)

    # System parameters
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train(
        N=args.N,
        d=args.d,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        r=args.r,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder_type=args.decoder_type,
        batch_size=args.batch_size,
        n_training_steps=args.n_training_steps,
        n_warmup_steps=args.n_warmup_steps,
        learning_rate=args.learning_rate,
        n_train_samples=args.n_train_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
