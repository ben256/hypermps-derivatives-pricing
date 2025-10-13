import argparse
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import matplotlib.pyplot as plt

from bs_functions import (
    sample_bs_params_batch,
    bs_call_price_analytical,
    bs_terminal_density,
    STRIKE_MIN,
    STRIKE_MAX,
)
from model import BenchmarkMPS
from utils import (
    seed_everything,
    create_recursive_folder,
    setup_logging,
    setup_device,
    bs_density_conditioning_param_len,
    idx_to_bits
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
        num_decoder_layers: int,
        batch_size: int,
        n_training_steps: int,
        n_warmup_steps: int,
        learning_rate: float,
        output_dir: str,
        seed: int,
    strikes_per_sample: int,
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

    conditioning_dim = bs_density_conditioning_param_len(d)

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
        num_decoder_layers=num_decoder_layers,
        decoder_type='shared'
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
    logger.info(f'  num_decoder_layers: {num_decoder_layers}')
    logger.info(f'  batch_size: {batch_size}')
    logger.info(f'  n_training_steps: {n_training_steps}')
    logger.info(f'  n_warmup_steps: {n_warmup_steps}')
    logger.info(f'  learning_rate: {learning_rate}')
    logger.info(f'  strikes_per_sample: {strikes_per_sample}')

    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    logger.info('Starting model training')

    if strikes_per_sample < 1:
        raise ValueError('strikes_per_sample must be at least 1')

    step = 0
    model.train()

    rolling_window = 300
    loss_buffer = deque(maxlen=rolling_window)

    indices_full_1d = torch.arange(N, device=device, dtype=torch.long)

    while step < n_training_steps:
        conditioning_params, params_list = sample_bs_params_batch(batch_size, rng)
        conditioning_params = conditioning_params.to(device=device, dtype=torch.float32)

        indices_full_nd = indices_full_1d.view(1, N, 1).expand(batch_size, -1, d)
        bits_full = idx_to_bits(indices_full_nd, d, L, gray=False)

        log_densities = model.forward(conditioning_params, bits_full)

        dx = grid[1] - grid[0]
        log_z = torch.logsumexp(log_densities + torch.log(dx), dim=1, keepdim=True)
        log_pdf = log_densities - log_z
        pdf = torch.exp(log_pdf)

        strikes = torch.empty(batch_size, strikes_per_sample, dtype=torch.float32)
        target_prices = torch.empty(batch_size, strikes_per_sample, dtype=torch.float32)

        for b in range(batch_size):
            params = params_list[b]
            strike_vector = torch.empty(strikes_per_sample, dtype=torch.float32).uniform_(STRIKE_MIN, STRIKE_MAX, generator=rng)
            strike_vector[0] = params['K']

            strikes[b] = strike_vector
            strike_values = strike_vector.tolist()
            target_prices[b] = torch.tensor(
                [
                    bs_call_price_analytical(
                        params['S0'],
                        float(k),
                        params['r'],
                        params['sigma'],
                        params['T'],
                    )
                    for k in strike_values
                ],
                dtype=torch.float32
            )

        strikes = strikes.to(device)
        target_prices = target_prices.to(device)

        payoff = torch.clamp(grid.view(1, 1, -1) - strikes.unsqueeze(-1), min=0.0)
        rates = conditioning_params[:, 1:2]
        maturities = conditioning_params[:, 3:4]
        discounted = torch.exp(-rates * maturities)
        predicted_prices = torch.sum(pdf.unsqueeze(1) * payoff, dim=2) * dx * discounted

        price_loss = F.smooth_l1_loss(predicted_prices, target_prices)
        loss = price_loss

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

        loss_buffer.append(loss.detach().item())
        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = sum(loss_buffer) / max(1, len(loss_buffer))
            logger.info(
                f'step {step}/{n_training_steps} - loss {loss.item():.6f} '
                f'({rolling_window}-avg {avg_loss:.6f}) - price_loss {price_loss.item():.6f} '
                f'- lr {current_lr:.2e}'
            )

        if step % 1000 == 0:
            logger.info('Generating plots')

            model.eval()
            with torch.no_grad():
                conditioning_params, params_list = sample_bs_params_batch(3, rng)
                conditioning_params = conditioning_params.to(device=device, dtype=torch.float32)

                plot_indices = indices_full_1d.view(1, N, 1).expand(3, -1, d)
                plot_bits = idx_to_bits(plot_indices, d, L, gray=False)

                learned_log_densities = model.forward(conditioning_params, plot_bits)

                fig, axs = plt.subplots(3, 1, figsize=(10, 12))

                for i in range(3):
                    b_params = params_list[i]
                    S0, r, sigma, T = b_params['S0'], b_params['r'], b_params['sigma'], b_params['T']

                    analytical_log_density = bs_terminal_density(grid, S0, r, sigma, T)
                    learned_log_density_corrected = learned_log_densities[i, :]

                    axs[i].plot(grid.cpu().numpy(), analytical_log_density.cpu().numpy(), label='Analytical PDF', color='blue', linewidth=2)
                    axs[i].plot(grid.cpu().numpy(), learned_log_density_corrected.cpu().numpy(), label='Learned PDF', color='orange', linestyle='--', linewidth=2)
                    axs[i].axvline(S0, color='gray', linestyle=':', label=f'S0={S0:.0f}')

                    axs[i].set_title(f'Terminal Distribution (S0={S0:.0f}, r={r:.2f}, σ={sigma:.2f}, T={T:.1f})')
                    axs[i].set_xlabel('Stock Price at Maturity (S_T)')
                    axs[i].set_ylabel('Log-Probability Density')
                    axs[i].legend()
                    axs[i].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'{output_dir}/density_comparison_step_{step}.png', dpi=150)
                plt.close(fig)
                logger.info(f'Saved density comparison plot to {output_dir}/density_comparison_step_{step}.png')

            model.train()

        step += 1

    logger.info('Training completed')

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
            'num_decoder_layers': num_decoder_layers,
        },
    }, model_path)
    logger.info(f'Model saved to {model_path}')

    logger.info('Benchmark training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Grid and problem parameters
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--grid-min', type=float, default=0.1)
    parser.add_argument('--grid-max', type=float, default=200.0)

    # Model architecture
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num-decoder-layers', type=int, default=1)

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-training-steps', type=int, default=20000)
    parser.add_argument('--n-warmup-steps', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--strikes-per-sample', type=int, default=32)

    # System parameters
    parser.add_argument('--output-dir', type=str, default='./output')
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
        num_decoder_layers=args.num_decoder_layers,
        batch_size=args.batch_size,
        n_training_steps=args.n_training_steps,
        n_warmup_steps=args.n_warmup_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        seed=args.seed,
        strikes_per_sample=args.strikes_per_sample,
    )
