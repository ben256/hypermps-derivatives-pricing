import argparse
import math
from collections import deque
from typing import cast

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
import matplotlib.pyplot as plt

from bs_functions import sample_bs_params_batch, bs_call_price_analytical, build_payoff_qtt, standard_normal_pdf, \
    build_put_payoff_qtt, bs_put_price_analytical
from benchmark import BenchmarkMPS
from qtt_functions import QTT, normalize_density_qtt, price_call_tt, make_z_grid, qtt_to_dense_vector
from utils import seed_everything, create_recursive_folder, setup_logging, setup_device, bs_density_conditioning_param_len


def train(
    N, d, grid_min, grid_max, rank, hidden_size, num_layers, num_decoder_layers,
    batch_size, n_training_steps, n_warmup_steps, learning_rate, output_dir, seed, decoder_type, log_space_loss,
    lambda_tv=0.0, lambda_entropy=0.0, lambda_kl=0.0, use_core_softmax=False, use_bit_mixing=False, bit_mix_init=0.1,
    multi_strike=0, lambda_moments=0.0, multi_strike_z_span: float = 3.0):
    seed_everything(seed)
    rng = torch.Generator()
    rng.manual_seed(seed)

    output_dir = create_recursive_folder(output_dir, 'benchmark')
    logger = setup_logging(output_dir)
    device = setup_device()

    logger.info('bit mixing test')

    logger.info('Initialising benchmark model training')
    logger.info(f'Using device: {device}')
    logger.info(f'Output directory: {output_dir}')

    conditioning_dim = bs_density_conditioning_param_len(d)
    L = int(math.log2(N))
    z_grid, dz = make_z_grid(grid_min, grid_max, L, device=device, dtype=torch.float32)
    volume_element = dz.item()

    model = BenchmarkMPS(
        N=N, r=rank, d=d, conditioning_dim=conditioning_dim,
        hidden_size=hidden_size, num_layers=num_layers, num_decoder_layers=num_decoder_layers,
        decoder_type=decoder_type, use_core_softmax=use_core_softmax,
        use_bit_mixing=use_bit_mixing, bit_mix_init=bit_mix_init,
    ).to(device=device, dtype=torch.float32)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    warmup = LambdaLR(optimizer, lambda step: min((step + 1) / max(1, n_warmup_steps), 1.0))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, n_training_steps - n_warmup_steps), eta_min=3e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[n_warmup_steps])

    logger.info(f'Training parameters:')
    logger.info(f'  N={N}, d={d}, L={L}, r={rank}')
    logger.info(f'  hidden_size={hidden_size}, num_layers={num_layers}, num_decoder_layers={num_decoder_layers}')
    logger.info(f'  decoder_type={decoder_type}')
    logger.info(f'  log_space_loss={log_space_loss}')
    logger.info(f'  use_core_softmax={use_core_softmax}')
    logger.info(f'  use_bit_mixing={use_bit_mixing}, bit_mix_init={bit_mix_init}')
    logger.info(f'  lambda_tv={lambda_tv}, lambda_entropy={lambda_entropy}, lambda_kl={lambda_kl}')
    logger.info(f'  lambda_moments={lambda_moments}')
    logger.info(f'  multi_strike={multi_strike}, multi_strike_z_span={multi_strike_z_span}')
    logger.info(f'  batch_size={batch_size}, n_training_steps={n_training_steps}')
    logger.info(f'  learning_rate={learning_rate}, n_warmup_steps={n_warmup_steps}')
    logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    model.train()
    rolling_window = 100
    loss_buffer = deque(maxlen=rolling_window)

    for step in range(n_training_steps):
        conditioning_params, params_list = sample_bs_params_batch(batch_size, rng)
        conditioning_params = conditioning_params.to(device=device, dtype=torch.float32)

        prices_pred = []
        prices_target = []
        prices_weight = []
        reg_terms = []

        for b in range(batch_size):
            params = params_list[b]
            m = torch.tensor(params['m'], device=device, dtype=z_grid.dtype)
            s = torch.tensor(params['s'], device=device, dtype=z_grid.dtype)

            qtt_p = cast(QTT, model(conditioning_params[b:b+1, :]))
            qtt_p = normalize_density_qtt(qtt_p, volume_element)

            # Optional density regularization in z-space
            if lambda_tv > 0.0 or lambda_entropy > 0.0 or lambda_kl > 0.0 or lambda_moments > 0.0:
                p_vec = qtt_to_dense_vector(qtt_p).clamp_min(1e-12)
                p_vec = p_vec / (p_vec.sum() * volume_element)
                tv = (p_vec[1:] - p_vec[:-1]).abs().mean()
                # entropy = -sum p log p (integral)
                entropy = -(p_vec * p_vec.log()).sum() * volume_element
                if lambda_kl > 0.0:
                    q = standard_normal_pdf(z_grid).clamp_min(1e-12)
                    q = q / (q.sum() * volume_element)
                    kl = ((p_vec * (p_vec.log() - q.log())) * volume_element).sum()
                else:
                    kl = p_vec.new_tensor(0.0)
                # moments of z
                if lambda_moments > 0.0:
                    mu = (z_grid * p_vec).sum() * volume_element
                    var = (((z_grid - mu) ** 2) * p_vec).sum() * volume_element
                    moment_pen = mu.pow(2) + (var.sqrt() - 1.0).pow(2)
                else:
                    moment_pen = p_vec.new_tensor(0.0)
                reg_terms.append(lambda_tv * tv + lambda_entropy * (-entropy) + lambda_kl * kl + lambda_moments * moment_pen)

            # S0 = params['S0']
            # K = params['K']
            #
            # is_call = K >= S0
            #
            # if is_call:
            #     payoff_tt = build_payoff_qtt(z_grid, dz, K, m, s, L)
            #     prices_pred.append(price_call_tt(qtt_p, payoff_tt, params['r'], params['T']))
            #     prices_target.append(bs_call_price_analytical(S0, K, params['r'], params['sigma'], params['T']))
            # else:
            #     payoff_tt = build_put_payoff_qtt(z_grid, dz, K, m, s, L)
            #     prices_pred.append(price_call_tt(qtt_p, payoff_tt, params['r'], params['T']))
            #     prices_target.append(bs_put_price_analytical(S0, K, params['r'], params['sigma'], params['T']))
            #
            # # Optional additional call strikes supervision per sample
            # if multi_strike and is_call:
            #     s_grid = torch.exp(m + s * z_grid)
            #     s_min = max(float(s_grid.detach().cpu()[0]), 25.0)
            #     s_max = min(float(s_grid.detach().cpu()[-1]), 250.0)
            #     extra_K = torch.linspace(s_min, s_max, steps=multi_strike, device=device)
            #     for K_extra in extra_K.tolist():
            #         payoff_extra = build_payoff_qtt(z_grid, dz, K_extra, m, s, L)
            #         prices_pred.append(price_call_tt(qtt_p, payoff_extra, params['r'], params['T']))
            #         prices_target.append(bs_call_price_analytical(S0, K_extra, params['r'], params['sigma'], params['T']))

            S0 = params['S0']
            K = params['K']

            if multi_strike:
                # sample strikes uniformly in z (log-moneyness) within +/- multi_strike_z_span
                z_lo = max(-multi_strike_z_span, float(z_grid[0]))
                z_hi = min( multi_strike_z_span, float(z_grid[-1]))
                z_strikes = torch.linspace(z_lo, z_hi, steps=int(multi_strike), device=device, dtype=z_grid.dtype)
                K_extras = torch.exp(m + s * z_strikes)
                # weight by standard normal pdf in z to avoid overemphasizing deep OTM/ITM
                w = standard_normal_pdf(z_strikes)
                w = (w / (w.mean() + 1e-12)).tolist()
                for K_extra, w_i in zip(K_extras.tolist(), w):
                    payoff_extra = build_payoff_qtt(z_grid, dz, K_extra, m, s, L)
                    prices_pred.append(price_call_tt(qtt_p, payoff_extra, params['r'], params['T']))
                    prices_target.append(bs_call_price_analytical(S0, K_extra, params['r'], params['sigma'], params['T']))
                    prices_weight.append(torch.tensor(w_i, device=device, dtype=z_grid.dtype))
            else:
                payoff_tt = build_payoff_qtt(z_grid, dz, K, m, s, L)
                prices_pred.append(price_call_tt(qtt_p, payoff_tt, params['r'], params['T']))
                prices_target.append(bs_call_price_analytical(S0, K, params['r'], params['sigma'], params['T']))
                prices_weight.append(torch.tensor(1.0, device=device, dtype=z_grid.dtype))

        pred = torch.stack(prices_pred)
        target = torch.tensor(prices_target, device=pred.device, dtype=pred.dtype)
        weight = torch.stack(prices_weight) if prices_weight else torch.ones_like(pred)

        if log_space_loss:
            # Log-space loss for scale-invariance
            log_pred = torch.log(pred + 1e-8)
            log_target = torch.log(target + 1e-8)
            huber = F.smooth_l1_loss(log_pred, log_target, reduction='none')
            loss = (weight * huber).mean() / (weight.mean() + 1e-12)
        else:
            # Price-space loss
            huber = F.smooth_l1_loss(pred, target, reduction='none')
            loss = (weight * huber).mean() / (weight.mean() + 1e-12)

        if reg_terms:
            loss = loss + torch.stack(reg_terms).mean()

        if not torch.isfinite(loss):
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = None
            optimizer.zero_grad(set_to_none=True)
            for g in optimizer.param_groups:
                g['lr'] *= 0.5  # back off
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_buffer.append(loss.item())
        if step % 10 == 0:
            avg_loss = sum(loss_buffer) / len(loss_buffer)
            lr = optimizer.param_groups[0]['lr']
            price_mae = (pred - target).abs().mean().item()
            price_mape = ((pred - target).abs() / (target + 1e-8)).mean().item() * 100

            if log_space_loss:
                logger.info(f'step {step}/{n_training_steps} - log_loss {loss.item():.6f} ({rolling_window}-avg {avg_loss:.6f}) - '
                           f'price_MAE {price_mae:.4f} - price_MAPE {price_mape:.2f}% - lr {lr:.2e}')
            else:
                logger.info(f'step {step}/{n_training_steps} - price_loss {loss.item():.6f} ({rolling_window}-avg {avg_loss:.6f}) - '
                           f'price_MAE {price_mae:.4f} - price_MAPE {price_mape:.2f}% - lr {lr:.2e}')

        if step % 50 == 0:
            plot_results(model, rng, z_grid, dz, volume_element, L, device, output_dir, step, logger)

    logger.info('Training completed')

    model_path = f'{output_dir}/model_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'N': N, 'r': rank, 'd': d, 'conditioning_dim': conditioning_dim,
            'hidden_size': hidden_size, 'num_layers': num_layers, 'num_decoder_layers': num_decoder_layers,
            'decoder_type': decoder_type,
        },
    }, model_path)
    logger.info(f'Model saved to {model_path}')
    plot_results(model, rng, z_grid, dz, volume_element, L, device, output_dir, n_training_steps, logger)


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
            r = torch.tensor(params['r'], device=device, dtype=z_grid.dtype)
            T = torch.tensor(params['T'], device=device, dtype=z_grid.dtype)

            qtt_p = cast(QTT, model(conditioning_params[i:i+1, :]))
            qtt_p = normalize_density_qtt(qtt_p, volume_element)
            learned_density = qtt_to_dense_vector(qtt_p)
            analytical_density = standard_normal_pdf(z_grid)

            # Also compute model-implied density via Breeden-Litzenberger
            with torch.no_grad():
                s_grid_dense = torch.exp(m + s * z_grid)
                K_bl = torch.linspace(float(s_grid_dense[0]), float(s_grid_dense[-1]), steps=200, device=device)
                C_vals = []
                for Kb in K_bl.tolist():
                    payoff_tt = build_payoff_qtt(z_grid, dz, Kb, m, s, L)
                    C_vals.append(price_call_tt(qtt_p, payoff_tt, params['r'], params['T']).item())
                C = torch.tensor(C_vals, device=device, dtype=z_grid.dtype)
                dK = K_bl[1] - K_bl[0]
                # central differences for second derivative
                second = (C[2:] - 2*C[1:-1] + C[:-2]) / (dK**2)
                fS = (second * torch.exp(r * T))  # risk-neutral density over S
                # interpolate fS onto S(z)
                Sz = s_grid_dense
                # simple linear interpolation in torch via numpy for brevity
                fS_np = fS.detach().cpu().numpy()
                K_mid = K_bl[1:-1].detach().cpu().numpy()
                Sz_np = Sz.detach().cpu().numpy()
                import numpy as _np
                fS_on_Sz = _np.interp(Sz_np, K_mid, fS_np)
                fS_on_Sz = torch.tensor(fS_on_Sz, device=device, dtype=z_grid.dtype)
                implied_density_z = (fS_on_Sz * s * Sz).clamp_min(0)
                # normalize implied z density for comparability
                implied_density_z = implied_density_z / (implied_density_z.sum() * dz)

            ax_density = axs[i, 0]
            ax_density.plot(z_grid.cpu().numpy(), analytical_density.cpu().numpy(), label='Analytical', color='blue', linewidth=2)
            ax_density.plot(z_grid.cpu().numpy(), learned_density.cpu().numpy(), label='Model', color='orange', linestyle='--', linewidth=2)
            ax_density.plot(z_grid.cpu().numpy(), implied_density_z.cpu().numpy(), label='Implied via C(K)', color='green', linestyle=':', linewidth=2)
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
    parser = argparse.ArgumentParser(description='Train benchmark MPS model for Black-Scholes')
    parser.add_argument('--N', type=int, default=256, help='Grid size (power of 2)')
    parser.add_argument('--d', type=int, default=1, help='Dimension')
    parser.add_argument('--grid-min', type=float, default=-8.0, help='Min z-value')
    parser.add_argument('--grid-max', type=float, default=8.0, help='Max z-value')
    parser.add_argument('--r', type=int, default=25, help='TT rank')
    parser.add_argument('--hidden-size', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of trunk layers')
    parser.add_argument('--num-decoder-layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--decoder-type', type=str, default='dimension', choices=['core', 'dimension', 'shared'])
    parser.add_argument('--log-space-loss', type=bool, default=True, help='Use log-space loss for scale-invariance')
    parser.add_argument('--use-core-softmax', type=bool, default=True, help='Normalize each core along bit dimension via softmax')
    parser.add_argument('--use-bit-mixing', type=bool, default=True, help='Enable small learned mixing between bit slices of each core')
    parser.add_argument('--bit-mix-init', type=float, default=0.4, help='Initial mixing strength in [0,0.5]')

    # parser.add_argument('--lambda-tv', type=float, default=0.01, help='Total variation regularization weight on density')
    # parser.add_argument('--lambda-entropy', type=float, default=0.001, help='Negative entropy penalty weight (encourage spread)')
    # parser.add_argument('--lambda-kl', type=float, default=0.1, help='KL(p||N(0,1)) weight in z-space')
    # parser.add_argument('--lambda-moments', type=float, default=0.1, help='Moment matching weight for z mean/vol')

    parser.add_argument('--lambda-tv', type=float, default=0.0, help='Total variation regularization weight on density')
    parser.add_argument('--lambda-entropy', type=float, default=0.0, help='Negative entropy penalty weight (encourage spread)')
    parser.add_argument('--lambda-kl', type=float, default=0.0, help='KL(p||N(0,1)) weight in z-space')
    parser.add_argument('--lambda-moments', type=float, default=0.0, help='Moment matching weight for z mean/vol')

    parser.add_argument('--multi-strike', type=int, default=0, help='Extra number of evenly spaced call strikes per sample')
    parser.add_argument('--multi-strike-z-span', type=float, default=3.0, help='Half-width in z for multi-strike sampling (uniform in z)')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--n-training-steps', type=int, default=1000, help='Training steps')
    parser.add_argument('--n-warmup-steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    train(args.N, args.d, args.grid_min, args.grid_max, args.r, args.hidden_size, args.num_layers,
        args.num_decoder_layers, args.batch_size, args.n_training_steps, args.n_warmup_steps,
        args.learning_rate, args.output_dir, args.seed, args.decoder_type, args.log_space_loss,
        args.lambda_tv, args.lambda_entropy, args.lambda_kl, args.use_core_softmax, args.multi_strike, args.lambda_moments,
            args.multi_strike_z_span, args.use_bit_mixing, args.bit_mix_init)
