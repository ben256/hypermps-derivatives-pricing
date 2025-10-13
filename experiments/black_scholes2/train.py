import argparse
import math
from collections import deque

import torch
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
from model import HyperHyperNetwork
from utils import (
    seed_everything,
    create_recursive_folder,
    setup_logging,
    setup_device,
    bs_density_conditioning_param_len,
    idx_to_bits,
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
        use_gray_bits: bool,
        strikes_per_sample: int,
):
    seed_everything(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    output_dir = create_recursive_folder(output_dir, f'hyper_training')
    logger = setup_logging(output_dir)
    device = setup_device()

    logger.info('Initialising Black-Scholes training')
    logger.info(f'Using device: {device}')
    logger.info(f'Output directory: {output_dir}')

    conditioning_dim = bs_density_conditioning_param_len(d)

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
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_training_steps - n_warmup_steps), eta_min=3e-5)
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
    logger.info(f'  use_gray_bits: {use_gray_bits}')
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
        bits_full = idx_to_bits(indices_full_nd, d, L, gray=use_gray_bits)

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

        price_loss = torch.nn.functional.mse_loss(predicted_prices, target_prices)
        reg_loss = model.orth_loss()
        loss = price_loss + reg_loss

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

        # Logging
        loss_buffer.append(loss.detach().item())
        if step % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss = sum(loss_buffer) / max(1, len(loss_buffer))
            logger.info(
                f'step {step}/{n_training_steps} - loss {loss.item():.6f} '
                f'({rolling_window}-avg {avg_loss:.6f}) - price_mse {price_loss.item():.6f} '
                f'- lr {current_lr:.2e}'
            )
        
        if step % 1000 == 0:
            logger.info('Generating plots')

            model.eval()
            with torch.no_grad():
                conditioning_params, params_list = sample_bs_params_batch(3, rng)
                conditioning_params = conditioning_params.to(device=device, dtype=torch.float32)

                plot_indices = indices_full_1d.view(1, N, 1).expand(3, -1, d)
                plot_bits = idx_to_bits(plot_indices, d, L, gray=use_gray_bits)

                learned_log_densities = model.forward(conditioning_params, plot_bits)

                fig, axs = plt.subplots(3, 1, figsize=(10, 12))

                for i in range(3):
                    b_params = params_list[i]
                    S0, r, sigma, T = b_params['S0'], b_params['r'], b_params['sigma'], b_params['T']

                    # Analytical density
                    analytical_log_density = bs_terminal_density(grid, S0, r, sigma, T)
                    learned_log_density_corrected = learned_log_densities[i, :]

                    # Plot both on same axis
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

    logger.info('Training finished')
    torch.save(model.state_dict(), f'{output_dir}/model_final.pth')
    logger.info(f'Model saved to {output_dir}/model_final.pth')


def main():
    parser = argparse.ArgumentParser()

    # Grid parameters
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--grid-min', type=float, default=10.0)  # min stock price
    parser.add_argument('--grid-max', type=float, default=200.0)  # max stock price

    # Model parameters
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--M', type=int, default=8)  # hypernetwork
    parser.add_argument('--orth-penalty', type=float, default=1e-5)
    parser.add_argument('--embedding-dim', type=int, default=64)  # hypernetwork
    parser.add_argument('--hidden-dim', type=int, default=128)  # hypernetwork/benchmark
    parser.add_argument('--n-layers', type=int, default=2)  # hypernetwork/benchmark
    parser.add_argument('--n-heads', type=int, default=4)  # hypernetwork
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--conditioning-tokens', type=int, default=4)  # hypernetwork

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--n-training-steps', type=int, default=30000)
    parser.add_argument('--n-warmup-steps', type=int, default=3000)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--n-train-samples', type=int, default=2048)
    parser.add_argument('--use-gray-bits', type=bool, default=True)
    parser.add_argument('--strikes-per-sample', type=int, default=32)

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
        seed=args.seed,
        use_gray_bits=args.use_gray_bits,
        strikes_per_sample=args.strikes_per_sample,
    )


if __name__ == '__main__':
    main()
