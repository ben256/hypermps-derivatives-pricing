import json
import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from model import QTTGenerator
from utils import idx_nd_to_bits, idx_nd_to_flat, sample_mixture_params_batch, flat_to_idx_nd, mixture_oracle_ytrue


def _select_validation_indices(N: int, d: int, device: torch.device, max_points: int) -> torch.Tensor:
    total = N ** d
    if total <= max_points:
        return torch.arange(total, device=device, dtype=torch.long)  # [S]
    perm = torch.randperm(total, device=device)
    return perm[:max_points]  # [S]


@torch.no_grad()
def evaluate_on_dataset(
        model: QTTGenerator,
        val_loader: DataLoader,
        d: int,
        N: int,
        grid: torch.Tensor,
        output_space: str,
        out_dir: str,
        device: torch.device,
        step_tag: str,
        max_points: int,
        plot_slices: bool,
        logger: logging.Logger,
) -> dict:
    model.eval()
    sum_se = 0.0
    sum_ae = 0.0
    sum_var = 0.0
    total_elems = 0
    first_batch_done = False

    for params, targets_full in val_loader:
        params = params.to(device, dtype=torch.float32)         # [B,cond_dim]
        targets_full = targets_full.to(device, dtype=torch.float32)  # [B,N^d]
        B = params.size(0)

        idx_flat_sel = _select_validation_indices(N, d, device, max_points)  # [S]
        idx_nd_sel = flat_to_idx_nd(idx_flat_sel, d, N)                      # [S,d]
        idx_nd_sel = idx_nd_sel.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B,S,d]
        bits = idx_nd_to_bits(idx_nd_sel, d=d, N=N)                          # [B,S,K]

        pred = model.forward_sampled(params, bits)                            # [B,S]
        y_true = targets_full.gather(1, idx_flat_sel.unsqueeze(0).expand(B, -1))  # [B,S]

        diff = pred - y_true
        sum_se += (diff.pow(2)).sum().item()
        sum_ae += (diff.abs()).sum().item()
        # per-item variance of y_true
        y_mean = y_true.mean(dim=1, keepdim=True)
        sum_var += ((y_true - y_mean).pow(2)).sum().item()
        total_elems += B * y_true.size(1)

        # plots on first batch
        if plot_slices and not first_batch_done:
            _plot_slices(
                model, params[0:1], targets_full[0], d, N, grid, output_space,
                out_dir, step_tag, device
            )
            first_batch_done = True

    mse = sum_se / max(1, total_elems)
    mae = sum_ae / max(1, total_elems)
    r2 = 1.0 - (sum_se / max(1e-12, sum_var)) if sum_var > 0 else float('nan')
    metrics = {"mse": mse, "mae": mae, "r2": r2}
    logger.info(f"[val:{step_tag}] mse={mse:.6f} mae={mae:.6f} r2={r2:.6f}")
    _save_metrics(metrics, out_dir, f"metrics_{step_tag}.json")
    model.train()
    return metrics


@torch.no_grad()
def evaluate_with_oracle(
        model: QTTGenerator,
        d: int,
        N: int,
        n_components: int,
        rng: np.random.Generator,
        grid: torch.Tensor,
        output_space: str,
        out_dir: str,
        device: torch.device,
        step_tag: str,
        max_points: int,
        plot_slices: bool,
        batch_size: int,
        logger: logging.Logger,
) -> dict:
    model.eval()
    sum_se = 0.0
    sum_ae = 0.0
    sum_var = 0.0
    total_elems = 0
    first_batch_done = False

    # Use a few oracle batches to estimate metrics
    num_val_batches = 4
    for vb in range(num_val_batches):
        B = batch_size
        cond_params, params_list = sample_mixture_params_batch(B, d, n_components, rng)
        params = cond_params.to(device, dtype=torch.float32)  # [B,cond_dim]

        idx_flat_sel = _select_validation_indices(N, d, device, max_points)  # [S]
        idx_nd_sel = flat_to_idx_nd(idx_flat_sel, d, N)  # [S,d]
        idx_nd_b = idx_nd_sel.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B,S,d]
        bits = idx_nd_to_bits(idx_nd_b, d=d, N=N)  # [B,S,K]

        pred = model.forward_sampled(params, bits)  # [B,S]
        y_true = mixture_oracle_ytrue(idx_nd_b, grid, params_list, output_space=output_space)  # [B,S]

        diff = pred - y_true
        sum_se += (diff.pow(2)).sum().item()
        sum_ae += (diff.abs()).sum().item()
        y_mean = y_true.mean(dim=1, keepdim=True)
        sum_var += ((y_true - y_mean).pow(2)).sum().item()
        total_elems += B * y_true.size(1)

        if plot_slices and not first_batch_done:
            # slice plots using b=0
            _plot_slices_oracle(
                model, params[0:1], params_list[0], d, N, grid, output_space,
                out_dir, step_tag, device
            )
            first_batch_done = True

    mse = sum_se / max(1, total_elems)
    mae = sum_ae / max(1, total_elems)
    r2 = 1.0 - (sum_se / max(1e-12, sum_var)) if sum_var > 0 else float('nan')
    metrics = {"mse": mse, "mae": mae, "r2": r2}
    logger.info(f"[val-oracle:{step_tag}] mse={mse:.6f} mae={mae:.6f} r2={r2:.6f}")
    _save_metrics(metrics, out_dir, f"metrics_{step_tag}.json")
    model.train()
    return metrics


def _save_metrics(metrics: dict, out_dir: str, file_name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


@torch.no_grad()
def _plot_slices(
        model: QTTGenerator,
        params_b1: torch.Tensor,          # [1,cond_dim]
        targets_full_b1: torch.Tensor,    # [N^d]
        d: int,
        N: int,
        grid: torch.Tensor,
        output_space: str,
        out_dir: str,
        step_tag: str,
        device: torch.device,
):
    slices_dir = os.path.join(out_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    center = N // 2
    for j in range(d):
        # build indices for a slice varying dim j
        coords = []
        for dim in range(d):
            if dim == j:
                coords.append(torch.arange(N, device=device, dtype=torch.long))  # [N]
            else:
                coords.append(torch.full((N,), center, device=device, dtype=torch.long))
        idx_nd_slice = torch.stack(coords, dim=-1).unsqueeze(0)  # [1,N,d]
        bits = idx_nd_to_bits(idx_nd_slice, d=d, N=N)            # [1,N,K]
        pred = model.forward_sampled(params_b1, bits).squeeze(0).detach().cpu().numpy()  # [N]

        idx_flat = idx_nd_to_flat(idx_nd_slice, N=N).squeeze(0)  # [N]
        y_true = targets_full_b1.gather(0, idx_flat).detach().cpu().numpy()

        # plot
        plt.figure(figsize=(6, 4))
        plt.plot(grid.detach().cpu().numpy(), y_true, label="target", lw=2)
        plt.plot(grid.detach().cpu().numpy(), pred, label="pred", lw=2)
        plt.xlabel("x")
        ylabel = "log-density" if output_space == "log" else "density"
        plt.ylabel(ylabel)
        plt.title(f"Slice dim={j} [{step_tag}]")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(slices_dir, f"slice_dim{j}_{step_tag}.png")
        plt.savefig(fname, dpi=150)
        plt.close()


@torch.no_grad()
def _plot_slices_oracle(
        model: QTTGenerator,
        params_b1: torch.Tensor,    # [1,cond_dim]
        oracle_params_b1: dict,
        d: int,
        N: int,
        grid: torch.Tensor,
        output_space: str,
        out_dir: str,
        step_tag: str,
        device: torch.device,
):
    slices_dir = os.path.join(out_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)
    center = N // 2
    for j in range(d):
        coords = []
        for dim in range(d):
            if dim == j:
                coords.append(torch.arange(N, device=device, dtype=torch.long))
            else:
                coords.append(torch.full((N,), center, device=device, dtype=torch.long))
        idx_nd_slice = torch.stack(coords, dim=-1).unsqueeze(0)  # [1,N,d]
        bits = idx_nd_to_bits(idx_nd_slice, d=d, N=N)  # [1,N,K]
        pred = model.forward_sampled(params_b1, bits).squeeze(0).detach().cpu().numpy()

        # oracle ground truth for slice
        y_true = mixture_oracle_ytrue(
            idx_nd_slice, grid, [oracle_params_b1], output_space=output_space
        ).squeeze(0).detach().cpu().numpy()

        plt.figure(figsize=(6, 4))
        plt.plot(grid.detach().cpu().numpy(), y_true, label="target", lw=2)
        plt.plot(grid.detach().cpu().numpy(), pred, label="pred", lw=2)
        plt.xlabel("x")
        ylabel = "log-density" if output_space == "log" else "density"
        plt.ylabel(ylabel)
        plt.title(f"Slice dim={j} [{step_tag}] (oracle)")
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(slices_dir, f"slice_dim{j}_{step_tag}.png")
        plt.savefig(fname, dpi=150)
        plt.close()