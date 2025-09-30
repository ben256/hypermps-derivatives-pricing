import argparse
import math
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TTDataset
from model import QTTGenerator
from eval import evaluate_on_dataset, evaluate_with_oracle
from utils import (
    find_dataset,
    create_recursive_folder,
    setup_logging,
    bits_to_idx_nd,
    idx_nd_to_flat,
    mixture_cond_len,
    sample_mixture_params_batch,
    mixture_oracle_ytrue,
)


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
        val_every: int = 1000,
        val_batch_size: int = None,
        val_max_points: int = 262144,
        save_plots: bool = True,
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

    # Data: training
    if use_dataset:
        try:
            # try to get val file too
            train_file, val_file, _, _ = find_dataset(dataset_dir, d=d, N=N, correlation=correlation)
        except Exception:
            train_file, val_file = None, None

        assert train_file is not None, "Training dataset file not found via find_dataset()."
        train_dataset = TTDataset(torch.load(train_file))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        sample_param, _ = next(iter(train_dataloader))
        cond_dim = sample_param.size(1)
    else:
        cond_dim = mixture_cond_len(d, n_components)
        train_dataloader = None
        train_file, val_file = None, None

    # Data: validation (prefer dataset)
    val_loader = None
    if val_batch_size is None:
        val_batch_size = batch_size
    if val_file is None and use_dataset:
        # reuse train as validation if no explicit val split
        val_file = train_file
    if val_file is not None and os.path.exists(val_file):
        val_dataset = TTDataset(torch.load(val_file))
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)

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

        if (step % max(1, val_every)) == 0 and step > 0:
            tag = f"step{step}"
            if val_loader is not None:
                evaluate_on_dataset(
                    model, val_loader, d, N, grid, output_space,
                    out_dir, device, tag, val_max_points, save_plots, logger
                )
            else:
                evaluate_with_oracle(
                    model, d, N, n_components, rng, grid, output_space,
                    out_dir, device, tag, val_max_points, save_plots, batch_size, logger
                )

    logger.info("Done")
    final_model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, final_model_path)

    # final validation
    tag = "final"
    if val_loader is not None:
        evaluate_on_dataset(
            model, val_loader, d, N, grid, output_space,
            out_dir, device, tag, val_max_points, True, logger
        )
    else:
        evaluate_with_oracle(
            model, d, N, n_components, rng, grid, output_space,
            out_dir, device, tag, val_max_points, True, batch_size, logger
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=2)
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

    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--val-batch-size', type=int, default=0)
    parser.add_argument('--val-max-points', type=int, default=262144)
    parser.add_argument('--save-plots', type=bool, default=True)
    args = parser.parse_args()

    val_bs = None if (args.val_batch_size is None or args.val_batch_size == 0) else args.val_batch_size

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
        seed=args.seed,
        val_every=args.val_every,
        val_batch_size=val_bs,
        val_max_points=args.val_max_points,
        save_plots=args.save_plots,
    )


if __name__ == '__main__':
    main()