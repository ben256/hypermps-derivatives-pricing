import argparse
import logging
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TTDataset
from process import sample_mixture_params, mixture_logpdf
from model import QTTGenerator
from utils import find_dataset, create_recursive_folder, setup_logging


def bits_to_idx_nd(bits: torch.Tensor, d: int, N: int) -> torch.Tensor:
    """
    bits: [B, S, K]
    returns idx_nd: [B, S, d]
    """
    B, S, K = bits.shape
    k = int(math.log2(N))
    # assert (1 << k) == N, "N must be power of 2"
    # assert K == d * k, f"Expected K=d*k={d*k}, got {K}"
    b = bits.to(torch.long).view(B, S, d, k)
    pow2 = (2 ** torch.arange(k - 1, -1, -1, device=b.device)).view(1, 1, 1, k)
    return (b * pow2).sum(dim=-1)  # [B, S, d]


def idx_nd_to_flat(idx_nd: torch.Tensor, N: int) -> torch.Tensor:
    """
    idx_nd: [B, S, d]
    """
    d = idx_nd.shape[-1]
    strides = torch.tensor([N ** (d - 1 - i) for i in range(d)],
                           device=idx_nd.device, dtype=torch.long)
    return (idx_nd * strides.view(1, 1, -1)).sum(dim=-1)


def mixture_cond_len(d: int, n_components: int) -> int:
    return n_components + n_components * d + n_components * (d * (d + 1) // 2)


def sample_mixture_params_batch(
        B: int, d: int, n_components: int, rng: np.random.Generator
):
    cond_dim = mixture_cond_len(d, n_components)
    cond = np.zeros((B, cond_dim), dtype=np.float32)
    params_list = []
    for b in range(B):
        weights, means, covs, invs, log_norms = sample_mixture_params(rng, d, n_components=n_components)
        parts = [weights.flatten(), means.flatten()]
        for cov in covs:
            parts.append(np.triu(cov)[np.triu_indices(d)])
        cond[b] = np.concatenate(parts).astype(np.float32)
        params_list.append({
            "weights": weights,
            "means": means,
            "invs": invs,
            "log_norms": log_norms,
        })
    return torch.from_numpy(cond), params_list


@torch.no_grad()
def mixture_oracle_ytrue(
        idx_nd: torch.Tensor,  # [B,S,d]
        grid: torch.Tensor,  # [N]
        params_list: list,  # len B, dict for each batch item
        output_space: str = "density",
) -> torch.Tensor:
    """
    y_true: [B,S].
    """
    device = idx_nd.device
    B, S, d = idx_nd.shape
    grid_np = grid.detach().cpu().numpy()
    outs = []
    for b in range(B):
        # Build X_b: [d,S] by indexing grid along each dim
        Xb = np.vstack([grid_np[idx_nd[b, :, j].detach().cpu().numpy()] for j in range(d)])
        p = params_list[b]
        logp = mixture_logpdf(Xb, p["weights"], p["means"], p["invs"], p["log_norms"])  # [S]
        if output_space == "log":
            y = torch.from_numpy(logp).to(device=device, dtype=torch.float32)
        else:
            y = torch.from_numpy(np.exp(logp)).to(device=device, dtype=torch.float32)
        outs.append(y)
    return torch.stack(outs, dim=0)  # [B,S]


def train_hypernetwork(
        d: int,
        N: int,
        correlation: float,
        basis_cores: int,
        max_rank: int,
        batch_size: int,
        learning_rate: float,
        steps: int,
        dataset_dir: str,
        output_dir: str,
        seed: int,
        num_samples: int = 2048,
        use_dataset: bool = False,
        n_components: int = 1,
        grid_min: float = -4.0,
        grid_max: float = 4.0,
        output_space: str = "log",
):
    """
    - Oracle: uses mixture oracle from process.py
    - Dataset mode: gathers y_true from dense targets
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    k = int(math.log2(N))
    assert (1 << k) == N, "N must be a power of 2"
    K = d * k
    S = num_samples
    grid = torch.linspace(grid_min, grid_max, N, device=device)

    if use_dataset:
        train_file, _, _, _ = find_dataset(dataset_dir, d=d, N=N, correlation=correlation)
        train_dataset = TTDataset(torch.load(train_file))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        sample_param, _ = next(iter(train_dataloader))
        cond_dim = sample_param.size(1)
    else:
        cond_dim = mixture_cond_len(d, n_components)
        train_dataloader = None

    model = QTTGenerator(
        d=d,
        N=N,
        r=max_rank,
        M=basis_cores,
        cond_dim=cond_dim,
        orth_penalty=1e-4
    ).to(device)

    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    out_dir = create_recursive_folder(output_dir, 'training_sampled')
    logger = setup_logging(out_dir)
    logger.info('Start sampled-contraction training')

    step = 0
    data_iter = iter(train_dataloader) if use_dataset else None

    while step < steps:
        if use_dataset:
            try:
                params, targets_full = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                params, targets_full = next(data_iter)

            params = params.to(device, dtype=torch.float32)  # [B, cond_dim]
            targets_full = targets_full.to(device, dtype=torch.float32)  # [B, N^d]
            B = params.size(0)

            bits = torch.randint(0, 2, (B, S, K), device=device, dtype=torch.long)
            pred = model.forward_sampled(params, bits)  # [B,S]

            idx_nd = bits_to_idx_nd(bits, d=d, N=N)  # [B,S,d]
            idx_flat = idx_nd_to_flat(idx_nd, N=N)  # [B,S]
            y_true = targets_full.gather(dim=1, index=idx_flat)  # [B,S]

        else:
            B = batch_size
            cond_params, params_list = sample_mixture_params_batch(B, d, n_components, rng)
            params = cond_params.to(device, dtype=torch.float32)  # [B, cond_dim]

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

    logger.info("Done")
    # try:
    #     if d == 1 and not use_dataset:
    #         export_dense_plots_1d(
    #             model=model,
    #             N=N,
    #             n_components=n_components,
    #             grid_min=grid_min,
    #             grid_max=grid_max,
    #             output_space=output_space,
    #             device=device,
    #             rng=rng,
    #             out_dir=out_dir,
    #             num_examples=4,
    #         )
    #         logger.info(f"Saved dense recon plots to {out_dir}/eval_plots")
    # except Exception as e:
    #     logger.error(f"Dense export failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=1)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--correlation', type=float, default=0.5)
    parser.add_argument('--max-rank', type=int, default=10)
    parser.add_argument('--basis-cores', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--num-samples', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--use-dataset', type=bool, default=False)
    parser.add_argument('--n-components', type=int, default=1)
    parser.add_argument('--grid-min', type=float, default=-4.0)
    parser.add_argument('--grid-max', type=float, default=4.0)
    parser.add_argument('--output-space', type=str, choices=['log', 'density'], default='log')
    parser.add_argument('--dataset-dir', type=str, default='./datasets')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_hypernetwork(
        d=args.d,
        N=args.N,
        correlation=args.correlation,
        basis_cores=args.basis_cores,
        max_rank=args.max_rank,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        steps=args.steps,
        use_dataset=args.use_dataset,
        n_components=args.n_components,
        output_space=args.output_space,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed)


if __name__ == '__main__':
    main()