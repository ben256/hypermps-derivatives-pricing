import argparse
import math
import os
import logging
import random

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from matplotlib import pyplot as plt

from model import QTTGenerator
from utils import (
    bits_to_idx_nd,
    idx_nd_to_bits,
    flat_to_idx_nd,
    mixture_cond_len,
    sample_mixture_params_batch,
    mixture_oracle_ytrue,
    setup_logging,
    create_recursive_folder,
)


logger = logging.getLogger(__name__)


def select_validation_indices(N: int, d: int, device: torch.device, max_points: int) -> torch.Tensor:
    total = N ** d
    k = int(min(max_points, total))
    if total <= k:
        return torch.arange(total, device=device, dtype=torch.long)
    sel = random.sample(range(total), k)
    return torch.tensor(sel, dtype=torch.long, device=device)


def train_hypernetwork(
        d: int,
        N: int,
        basis_cores: int,
        max_rank: int,
        batch_size: int,
        learning_rate: float,
        steps: int,
        output_dir: str,
        seed: int,
        num_samples: int = 2048,
        n_components: int = 1,
        grid_min: float = -4.0,
        grid_max: float = 4.0,
        output_space: str = "log",
        val_every: int = 1000,
        val_max_points: int = 262144,
):

    output_dir = create_recursive_folder(output_dir, 'training')
    logger = setup_logging(output_dir)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rng = np.random.default_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k = int(math.log2(N))
    assert (1 << k) == N, "N must be a power of 2"
    K = d * k
    S = num_samples
    grid = torch.linspace(grid_min, grid_max, N, device=device)

    # conditioning dimension implied by oracle mixture params
    cond_dim = mixture_cond_len(d, n_components)

    model = QTTGenerator(
        d=d,
        N=N,
        r=max_rank,
        M=basis_cores,
        cond_dim=cond_dim,
        orth_penalty=1e-4
    ).to(device)

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    logger.info(f"Starting training | d={d} N={N} k={k} K={K} S={S} max_rank={max_rank} basis_cores={basis_cores} "
                f"batch_size={batch_size} lr={learning_rate} steps={steps} n_components={n_components} "
                f"grid=[{grid_min},{grid_max}] output_space={output_space} device={device}")

    step = 0
    model.train()

    while step < steps:
        # Oracle batch
        B = batch_size
        cond_params, params_list = sample_mixture_params_batch(B, d, n_components, rng)
        params = cond_params.to(device, dtype=torch.float32)  # [B, conditional_dim]

        bits = torch.randint(0, 2, (B, S, K), device=device, dtype=torch.long)
        pred = model.forward_sampled(params, bits)  # [B,S]

        idx_nd = bits_to_idx_nd(bits, d=d, N=N)  # [B,S,d]
        y_true = mixture_oracle_ytrue(idx_nd, grid, params_list, output_space=output_space)

        loss = F.mse_loss(pred, y_true) + model.orth_loss()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        if (step % 250) == 0:
            logger.info(f"step {step}/{steps}  loss={loss.item():.6f}")
        step += 1

        if (step % max(1, val_every)) == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                sum_se = 0.0
                sum_ae = 0.0
                sum_var = 0.0
                total_elems = 0
                num_val_batches = 4
                for _ in range(num_val_batches):
                    Bv = batch_size
                    cond_params_v, params_list_v = sample_mixture_params_batch(Bv, d, n_components, rng)
                    params_v = cond_params_v.to(device, dtype=torch.float32)

                    idx_flat_sel = select_validation_indices(N, d, device, val_max_points)  # [Ssel]
                    idx_nd_sel = flat_to_idx_nd(idx_flat_sel, d, N)  # [Ssel,d]
                    idx_nd_b = idx_nd_sel.unsqueeze(0).expand(Bv, -1, -1).contiguous()  # [Bv,Ssel,d]
                    bits_v = idx_nd_to_bits(idx_nd_b, d=d, N=N)  # [Bv,Ssel,K]

                    pred_v = model.forward_sampled(params_v, bits_v)  # [Bv,Ssel]
                    y_true_v = mixture_oracle_ytrue(idx_nd_b, grid, params_list_v, output_space=output_space)  # [Bv,Ssel]

                    diff = pred_v - y_true_v
                    sum_se += (diff.pow(2)).sum().item()
                    sum_ae += (diff.abs()).sum().item()
                    y_mean = y_true_v.mean(dim=1, keepdim=True)
                    sum_var += ((y_true_v - y_mean).pow(2)).sum().item()
                    total_elems += Bv * y_true_v.size(1)

                mse = sum_se / max(1, total_elems)
                mae = sum_ae / max(1, total_elems)
                r2 = 1.0 - (sum_se / max(1e-12, sum_var)) if sum_var > 0 else float('nan')
                logger.info(f"[eval step {step}] mse={mse:.6f} mae={mae:.6f} r2={r2:.6f}")
            model.train()

    logger.info("Training complete. Saving model and final plots...")
    final_model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, final_model_path)

    # Final 4×d plot: 4 random samples (columns) × d dimensions (rows)
    model.eval()
    with torch.no_grad():
        cols = 4
        cond_params_plot, params_list_plot = sample_mixture_params_batch(cols, d, n_components, rng)
        params_plot = cond_params_plot.to(device, dtype=torch.float32)  # [4,conditional_dim]

        center = N // 2
        fig = plt.figure(figsize=(4 * cols, 3 * d))

        x_axis = grid.detach().cpu().numpy()
        ylabel = "log-density" if output_space == "log" else "density"

        for c in range(cols):
            for j in range(d):
                # build indices for a slice varying dimension j
                coords = []
                for dim in range(d):
                    if dim == j:
                        coords.append(torch.arange(N, device=device, dtype=torch.long))  # [N]
                    else:
                        coords.append(torch.full((N,), center, device=device, dtype=torch.long))
                idx_nd_slice = torch.stack(coords, dim=-1).unsqueeze(0)  # [1,N,d]
                bits_slice = idx_nd_to_bits(idx_nd_slice, d=d, N=N)  # [1,N,K]

                pred_slice = model.forward_sampled(params_plot[c:c+1], bits_slice).squeeze(0).detach().cpu().numpy()  # [N]
                y_true_slice = mixture_oracle_ytrue(
                    idx_nd_slice, grid, [params_list_plot[c]], output_space=output_space
                ).squeeze(0).detach().cpu().numpy()

                ax = fig.add_subplot(d, cols, j * cols + c + 1)
                ax.plot(x_axis, y_true_slice, label="target", lw=2)
                ax.plot(x_axis, pred_slice, label="pred", lw=2)
                if j == d - 1:
                    ax.set_xlabel("x")
                if c == 0:
                    ax.set_ylabel(ylabel)
                if j == 0:
                    ax.set_title(f"sample {c}")
        # only one legend for the whole figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout(rect=(0, 0, 0.98, 1))
        out_path = os.path.join(output_dir, f"final_4x{d}_slices.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    logger.info(f"Saved model to {final_model_path}")
    logger.info(f"Saved final plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--max-rank', type=int, default=15)
    parser.add_argument('--basis-cores', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-samples', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--n-components', type=int, default=1)
    parser.add_argument('--grid-min', type=float, default=-4.0)
    parser.add_argument('--grid-max', type=float, default=4.0)
    parser.add_argument('--output-space', type=str, choices=['log', 'density'], default='log')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--val-max-points', type=int, default=262144)
    args = parser.parse_args()

    train_hypernetwork(
        d=args.d,
        N=args.N,
        basis_cores=args.basis_cores,
        max_rank=args.max_rank,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        steps=args.steps,
        n_components=args.n_components,
        output_space=args.output_space,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        output_dir=args.output_dir,
        seed=args.seed,
        val_every=args.val_every,
        val_max_points=args.val_max_points,
    )


if __name__ == '__main__':
    main()