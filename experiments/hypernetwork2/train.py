import argparse
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from functions import sample_gaussian_params_batch, gaussian_analytical
from model import HyperHyperNetwork
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
    compute_grid_coverage_metrics,
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
        M: int,
        orth_penalty: float,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        conditioning_tokens: int,
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

    output_dir = create_recursive_folder(output_dir, 'training')
    logger = setup_logging(output_dir)
    device = setup_device()

    logger.info('Initialising training')
    logger.info(f'Using device: {device}')
    logger.info(f'Output directory: {output_dir}')

    conditioning_dim = gaussian_conditioning_param_len(d)

    L = int(math.log2(N))
    K = d * L
    grid = torch.linspace(grid_min, grid_max, N, device=device, dtype=torch.float32)

    model = HyperHyperNetwork(
        N=N,
        r=r,
        M=M,
        orth_penalty=orth_penalty,
        d=d,
        conditioning_dim=conditioning_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        conditioning_tokens=conditioning_tokens,
    ).to(device=device, dtype=torch.float32)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    warmup = LambdaLR(optimizer, lambda step: min((step + 1) / max(1, n_warmup_steps), 1.0))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_training_steps - n_warmup_steps), eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[n_warmup_steps])

    logger.info(f'Training parameters:')
    logger.info(f'  N: {N}')
    logger.info(f'  d: {d}')
    logger.info(f'  r: {r}')
    logger.info(f'  M: {M}')
    logger.info(f'  orth_penalty: {orth_penalty}')
    logger.info(f'  embedding_dim: {embedding_dim}')
    logger.info(f'  hidden_dim: {hidden_dim}')
    logger.info(f'  n_layers: {n_layers}')
    logger.info(f'  n_heads: {n_heads}')
    logger.info(f'  dropout: {dropout}')
    logger.info(f'  conditioning_tokens: {conditioning_tokens}')
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
        loss = loss + model.orth_loss()

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
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = loss_sum / len(loss_buffer)
            logger.info(f"Step {step}: loss = {loss_value:.6f}, avg_loss({rolling_window}) = {avg_loss:.6f}, lr = {current_lr:.2e}")

        if step % 2000 == 0 and step > 0:
            logger.info(f"Generating plots at step {step}...")
            plot_slices(model, grid, d, N, device, output_dir, step, rng, num_samples=4)
            logger.info(f"Plots saved to {output_dir}/slices_step_{step}.png")

    logger.info('Training complete')
    torch.save(model.state_dict(), f'{output_dir}/model_final.pth')
    logger.info(f'Model saved to {output_dir}/model_final.pth')

    # Evaluate the trained model
    logger.info('Computing evaluation metrics...')
    eval_metrics = compute_evaluation_metrics(
        model=model,
        grid=grid,
        d=d,
        N=N,
        device=device,
        rng=rng,
        n_test_samples=n_train_samples,
        n_test_distributions=100,
    )

    grid_metrics = compute_grid_coverage_metrics(
        model=model,
        grid=grid,
        d=d,
        N=N,
        device=device,
        rng=rng,
        n_distributions=10,
    )

    # Combine all metrics
    all_metrics = {**eval_metrics, **grid_metrics}

    # Save metrics to file
    save_metrics(all_metrics, f'{output_dir}/evaluation_metrics.json')

    # Print summary
    print_metrics_summary(all_metrics, title="Final Model Evaluation")

    logger.info('Evaluation complete')
    logger.info(f'Metrics saved to {output_dir}/evaluation_metrics.json')


def main():
    parser = argparse.ArgumentParser()
    # Grid parameters
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--grid-min', type=float, default=-4.0)
    parser.add_argument('--grid-max', type=float, default=4.0)

    # Model parameters
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--orth-penalty', type=float, default=1e-5)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--conditioning-tokens', type=int, default=8)

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-training-steps', type=int, default=20000)
    parser.add_argument('--n_warmup_steps', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--n_train_samples', type=int, default=512)

    # General parameters
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(
        N=args.N,
        d=args.d,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        r=args.r,
        M=args.M,
        orth_penalty=args.orth_penalty,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        conditioning_tokens=args.conditioning_tokens,
        batch_size=args.batch_size,
        n_training_steps=args.n_training_steps,
        n_warmup_steps=args.n_warmup_steps,
        learning_rate=args.learning_rate,
        n_train_samples=args.n_train_samples,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == '__main__':
    main()