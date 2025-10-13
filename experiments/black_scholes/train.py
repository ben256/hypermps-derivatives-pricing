import argparse
import math
from collections import deque

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import matplotlib.pyplot as plt

from bs_functions import sample_bs_params_batch, bs_call_price_analytical, build_payoff_qtt, standard_normal_pdf
from model import HyperHyperNetwork
from qtt_functions import QTT, normalize_density_qtt, price_call_tt, _qtt_to_tn, make_z_grid
from utils import (
    seed_everything,
    create_recursive_folder,
    setup_logging,
    setup_device,
    bs_density_conditioning_param_len,
)


def train(
        N, d, grid_min, grid_max, rank, M, orth_penalty, embedding_dim, hidden_dim, n_layers,
        n_heads, dropout, conditioning_tokens, num_decoder_layers,
        batch_size, n_training_steps, n_warmup_steps, learning_rate, output_dir, seed, 
        use_gray_bits, log_space_loss):
    seed_everything(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    output_dir = create_recursive_folder(output_dir, 'hyper')
    logger = setup_logging(output_dir)
    device = setup_device()

    logger.info('Initialising hypernetwork model training')
    logger.info(f'Using device: {device}')
    logger.info(f'Output directory: {output_dir}')

    conditioning_dim = bs_density_conditioning_param_len(d)
    L = int(math.log2(N))
    z_grid, dz = make_z_grid(grid_min, grid_max, L, device=device, dtype=torch.float32)
    volume_element = dz.item()

    model = HyperHyperNetwork(
        N=N, r=rank, M=M, orth_penalty=orth_penalty, d=d,
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
    logger.info(f'  N={N}, d={d}, L={L}, r={rank}')
    logger.info(f'  M={M}, orth_penalty={orth_penalty}')
    logger.info(f'  embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, n_layers={n_layers}')
    logger.info(f'  n_heads={n_heads}, dropout={dropout}, conditioning_tokens={conditioning_tokens}')
    logger.info(f'  use_gray_bits={use_gray_bits}')
    logger.info(f'  log_space_loss={log_space_loss}')
    logger.info(f'  batch_size={batch_size}, n_training_steps={n_training_steps}')
    logger.info(f'  learning_rate={learning_rate}, n_warmup_steps={n_warmup_steps}')
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    model.train()
    rolling_window = 300
    loss_buffer = deque(maxlen=rolling_window)

    for step in range(n_training_steps):
        conditioning_params, params_list = sample_bs_params_batch(batch_size, rng)
        conditioning_params = conditioning_params.to(device=device, dtype=torch.float32)

        prices_pred = []
        prices_target = []

        for b in range(batch_size):
            params = params_list[b]
            m = torch.tensor(params['m'], device=device, dtype=z_grid.dtype)
            s = torch.tensor(params['s'], device=device, dtype=z_grid.dtype)

            # Get QTT representation from hypernetwork
            u_batch, cores_batch, v_batch = model.forward(conditioning_params[b:b+1, :])
            # Convert from batch format to QTT format
            u = u_batch[0]  # [r]
            v = v_batch[0]  # [r]
            cores = [cores_batch[0, k, :, :, :] for k in range(cores_batch.shape[1])]  # List of [2, r, r]
            qtt_p = QTT(u=u, cores=cores, v=v)
            qtt_p = normalize_density_qtt(qtt_p, volume_element)

            payoff_tt = build_payoff_qtt(z_grid, dz, params['K'], m, s, L)
            prices_pred.append(price_call_tt(qtt_p, payoff_tt, params['r'], params['T']))
            prices_target.append(bs_call_price_analytical(params['S0'], params['K'], params['r'], params['sigma'], params['T']))

        pred = torch.stack(prices_pred)
        target = torch.tensor(prices_target, device=pred.device, dtype=pred.dtype)
        
        if log_space_loss:
            # Log-space loss for scale-invariance
            log_pred = torch.log(pred + 1e-8)
            log_target = torch.log(target + 1e-8)
            loss = F.mse_loss(log_pred, log_target) + model.orth_loss()
        else:
            # Price-space loss
            loss = F.mse_loss(pred, target) + model.orth_loss()

        if not torch.isfinite(loss):
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    logger.warning(f'Non-finite gradient detected in parameter: {p.shape}')
            logger.warning('Skipping update due to non-finite loss')
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_buffer.append(loss.item())
        if step % 25 == 0:
            avg_loss = sum(loss_buffer) / len(loss_buffer)
            lr = optimizer.param_groups[0]['lr']
            # Also compute price-space metrics for interpretability
            price_mae = (pred - target).abs().mean().item()
            price_mape = ((pred - target).abs() / (target + 1e-8)).mean().item() * 100
            
            if log_space_loss:
                logger.info(f'step {step}/{n_training_steps} - log_loss {loss.item():.6f} ({rolling_window}-avg {avg_loss:.6f}) - '
                           f'price_MAE {price_mae:.4f} - price_MAPE {price_mape:.2f}% - lr {lr:.2e}')
            else:
                logger.info(f'step {step}/{n_training_steps} - price_loss {loss.item():.6f} ({rolling_window}-avg {avg_loss:.6f}) - '
                           f'price_MAE {price_mae:.4f} - price_MAPE {price_mape:.2f}% - lr {lr:.2e}')

        if step % 100 == 0:
            plot_results(model, rng, z_grid, dz, volume_element, L, device, output_dir, step, logger)

    logger.info('Training completed')

    model_path = f'{output_dir}/model_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'N': N, 'r': rank, 'd': d, 'M': M,
            'conditioning_dim': conditioning_dim,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'dropout': dropout,
            'conditioning_tokens': conditioning_tokens,
        },
    }, model_path)
    logger.info(f'Model saved to {model_path}')


def plot_results(model, rng, z_grid, dz, volume_element, L, device, output_dir, step, logger):
    model.eval()
    with torch.no_grad():
        conditioning_params, params_list = sample_bs_params_batch(3, rng)
        conditioning_params = conditioning_params.to(device=device, dtype=torch.float32)
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))

        for i in range(3):
            params = params_list[i]
            m = torch.tensor(params['m'], device=device, dtype=z_grid.dtype)
            s = torch.tensor(params['s'], device=device, dtype=z_grid.dtype)

            # Get QTT from hypernetwork
            u_batch, cores_batch, v_batch = model.forward(conditioning_params[i:i+1, :])
            u = u_batch[0]
            v = v_batch[0]
            cores = [cores_batch[0, k, :, :, :] for k in range(cores_batch.shape[1])]
            qtt_p = QTT(u=u, cores=cores, v=v)
            qtt_p = normalize_density_qtt(qtt_p, volume_element)
            
            tn_density = _qtt_to_tn(qtt_p)
            learned_density = tn_density.torch().reshape(-1)
            analytical_density = standard_normal_pdf(z_grid)

            ax_density = axs[i, 0]
            ax_density.plot(z_grid.cpu().numpy(), analytical_density.cpu().numpy(), label='Analytical', color='blue', linewidth=2)
            ax_density.plot(z_grid.cpu().numpy(), learned_density.cpu().numpy(), label='Model', color='orange', linestyle='--', linewidth=2)
            ax_density.axvline(0.0, color='gray', linestyle=':', linewidth=1)
            ax_density.set_title(f'Density (S0={params["S0"]:.0f}, r={params["r"]:.2f}, σ={params["sigma"]:.2f}, T={params["T"]:.1f})')
            ax_density.set_xlabel('Standardized Log-Price (z)')
            ax_density.set_ylabel('Probability Density')
            ax_density.legend()
            ax_density.grid(True, alpha=0.3)

            s_grid = torch.exp(m + s * z_grid)
            s_min = max(float(s_grid.detach().cpu()[0]), 25.0)
            s_max = min(float(s_grid.detach().cpu()[-1]), 250.0)
            strike_vals = torch.linspace(s_min, s_max, steps=50)

            tt_prices = []
            analytic_prices = []
            for K_val in strike_vals.tolist():
                payoff_tt = build_payoff_qtt(z_grid, dz, K_val, m, s, L)
                tt_prices.append(price_call_tt(qtt_p, payoff_tt, params['r'], params['T']).item())
                analytic_prices.append(bs_call_price_analytical(params['S0'], K_val, params['r'], params['sigma'], params['T']))

            ax_price = axs[i, 1]
            ax_price.plot(strike_vals.numpy(), analytic_prices, label='Analytical', color='blue', linewidth=2)
            ax_price.plot(strike_vals.numpy(), tt_prices, label='Model', color='orange', linestyle='--', linewidth=2)

            K_sample = params['K']
            if s_min <= K_sample <= s_max:
                payoff_sample = build_payoff_qtt(z_grid, dz, K_sample, m, s, L)
                tt_sample = price_call_tt(qtt_p, payoff_sample, params['r'], params['T']).item()
                analytic_sample = bs_call_price_analytical(params['S0'], K_sample, params['r'], params['sigma'], params['T'])
                ax_price.scatter([K_sample], [analytic_sample], color='blue', marker='o', s=40, label='Analytical @ K')
                ax_price.scatter([K_sample], [tt_sample], color='orange', marker='x', s=40, label='Model @ K')

            ax_price.set_xlim(s_min, s_max)
            ax_price.set_xlabel('Strike (K)')
            ax_price.set_ylabel('Call Price')
            ax_price.set_title('Call Price Curve')
            ax_price.legend()
            ax_price.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/diagnostics_step_{step}.png', dpi=150)
        plt.close(fig)
        logger.info(f'Saved diagnostics to {output_dir}/diagnostics_step_{step}.png')

    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hypernetwork MPS model for Black-Scholes')
    parser.add_argument('--N', type=int, default=256, help='Grid size (power of 2)')
    parser.add_argument('--d', type=int, default=1, help='Dimension')
    parser.add_argument('--grid-min', type=float, default=-8.0, help='Min z-value')
    parser.add_argument('--grid-max', type=float, default=8.0, help='Max z-value')
    parser.add_argument('--r', type=int, default=16, help='TT rank')
    parser.add_argument('--M', type=int, default=4, help='Number of basis cores')
    parser.add_argument('--orth-penalty', type=float, default=1e-5, help='Orthogonality penalty')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--conditioning-tokens', type=int, default=4, help='Number of conditioning tokens')
    parser.add_argument('--num-decoder-layers', type=int, default=1, help='Number of decoder layers (unused for hypernetwork)')
    parser.add_argument('--use-gray-bits', type=bool, default=False, help='Use Gray code for bit encoding')
    parser.add_argument('--log-space-loss', type=bool, default=True, help='Use log-space loss for scale-invariance')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--n-training-steps', type=int, default=5000, help='Training steps')
    parser.add_argument('--n-warmup-steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    train(args.N, args.d, args.grid_min, args.grid_max, args.r, args.M, args.orth_penalty,
          args.embedding_dim, args.hidden_dim, args.n_layers, args.n_heads, args.dropout,
          args.conditioning_tokens, args.num_decoder_layers, args.batch_size, args.n_training_steps,
          args.n_warmup_steps, args.learning_rate, args.output_dir, args.seed,
          args.use_gray_bits, args.log_space_loss)
