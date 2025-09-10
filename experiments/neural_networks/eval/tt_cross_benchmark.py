import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import tntorch as tn
from torch import nn, optim
from torch.utils.data import DataLoader

from data_processing.dataset import TTDataset
from model.neural_mps import FNNNeuralMPS
from train.utils import create_recursive_folder, setup_logging, find_dataset


def eval_tt(tt_cores: List[torch.Tensor], x_indices: List[int]) -> torch.Tensor:
    """Contract a batch of TT cores at a chosen multi-index.

    Args:
        tt_cores: list length d, each tensor [batch, r_i, n, r_{i+1}]
        x_indices: list length d, each an int index into the physical dimension n

    Returns:
        Tensor [batch, 1, 1] (assuming boundary ranks are 1)
    """
    v = tt_cores[0][:, :, x_indices[0], :]
    for i in range(1, len(tt_cores)):
        v = v @ tt_cores[i][:, :, x_indices[i], :]
    return v


def reconstruct_diagonal(cores: List[torch.Tensor], N: int, d: int) -> torch.Tensor:
    """Reconstruct f(i,i,...,i) for i=0..N-1 -> [batch, N]."""
    batch = cores[0].shape[0]
    vals = []
    for i in range(N):
        v = eval_tt(cores, [i] * d)
        vals.append(v.view(batch))
    return torch.stack(vals, dim=1)


@dataclass
class EarlyStopping:
    patience: int = 8
    delta: float = 0.0
    offset: int = 5
    best_loss: Optional[float] = None
    counter: int = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stopped: bool = False

    def step(self, epoch: int, val_loss: float, model: nn.Module):
        if epoch < self.offset:
            return
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True


def build_tt_ranks(d: int, internal_rank: int) -> List[int]:
    if d == 1:
        return [1, 1]
    return [1] + [internal_rank] * (d - 1) + [1]


def train_fnn_model(
        model: FNNNeuralMPS,
        train_loader: DataLoader,
        val_loader: DataLoader,
        N: int,
        d: int,
        device: torch.device,
        epochs: int,
        lr: float,
        weight_decay: float,
        logger,
        early: EarlyStopping,
) -> Dict[str, List[float]]:
    crit = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_hist, val_hist = [], []
    for epoch in range(epochs):
        model.train()
        run = 0.0
        for params, target in train_loader:
            params = params.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)
            opt.zero_grad()
            cores = model(params)
            out = reconstruct_diagonal(cores, N, d)
            loss = crit(out, target)
            loss.backward()
            opt.step()
            run += loss.item()
        avg_train = run / max(1, len(train_loader))
        train_hist.append(avg_train)
        model.eval()
        val_run = 0.0
        with torch.no_grad():
            for params, target in val_loader:
                params = params.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)
                cores = model(params)
                out = reconstruct_diagonal(cores, N, d)
                val_run += crit(out, target).item()
        avg_val = val_run / max(1, len(val_loader))
        val_hist.append(avg_val)
        logger.info(f'[d={d}] Epoch {epoch+1}/{epochs} Train {avg_train:.6f} Val {avg_val:.6f}')
        early.step(epoch, avg_val, model)
        if early.stopped:
            logger.info(f'[d={d}] Early stopping at epoch {epoch+1}')
            break
    if early.best_state is not None:
        model.load_state_dict(early.best_state)
    return {'train': train_hist, 'val': val_hist}


def gaussian_target_function(x: np.ndarray, A: float, c: np.ndarray, cov: np.ndarray) -> torch.Tensor:
    c = c.reshape(-1, 1)
    cov_inv = np.linalg.inv(cov)
    diff = x - c
    exponent_term = -0.5 * np.einsum('ib,ij,jb->b', diff, cov_inv, diff)
    return torch.from_numpy(A * np.exp(exponent_term))


def function_wrapper(*ix, A, c, cov_matrix, N, device):
    d = len(ix)
    x_vec = []
    for i in range(d):
        indices = ix[i].cpu().numpy() if isinstance(ix[i], torch.Tensor) else ix[i]
        indices = indices.astype(int)
        x_vec.append(np.take(np.linspace(-1, 1, N), indices))
    out = gaussian_target_function(np.stack(x_vec), A, c, cov_matrix)
    return out.to(device)


def generate_covariance_matrix(rng: np.random.Generator, d: int, correlation: float) -> np.ndarray:
    stds = rng.uniform(0.1, 1.0, size=d)
    corr = np.full((d, d), correlation)
    np.fill_diagonal(corr, 1.0)
    # ensure PSD
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(corr)
        vals[vals < 0] = 0
        corr = vecs @ np.diag(vals) @ vecs.T
    S = np.diag(stds)
    return S @ corr @ S


def run_tt_cross_single_sample(d: int, N: int, max_rank: int, correlation: float, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.2, 1.0)
    c = rng.uniform(-0.5, 0.5, size=d)
    cov = generate_covariance_matrix(rng, d, correlation)
    domain = [torch.arange(N, device=device) for _ in range(d)]
    ranks = [max_rank] * (d - 1)
    tt_tensor = tn.cross(
        function=lambda *ix: function_wrapper(*ix, A=A, c=c, cov_matrix=cov, N=N, device=device),
        domain=domain,
        eps=1e-7,
        ranks_tt=ranks,
        max_iter=100,
        early_stopping_patience=3,
        early_stopping_tolerance=1e-8,
        verbose=False,
        suppress_warnings=True,
        device=device,
    )
    # tn.cross may sometimes return (tensor, info); handle gracefully
    if isinstance(tt_tensor, tuple):
        tt_tensor = tt_tensor[0]
    # Reconstruct diagonal values
    cores = [core.unsqueeze(0) for core in tt_tensor.cores]  # add batch dim=1 for reuse of eval
    diag_vals = reconstruct_diagonal(cores, N, d).squeeze(0)  # [N]
    # Ground truth analytic diagonal
    grid = np.linspace(-1, 1, N)
    x = np.stack([grid] * d)
    target = gaussian_target_function(x, A, c, cov).float()
    # errors
    mse = torch.mean((diag_vals - target) ** 2).item()
    mae = torch.mean(torch.abs(diag_vals - target)).item()
    return {'A': float(A), 'c': c.tolist(), 'mse': mse, 'mae': mae}


def benchmark(
        d_start: int,
        d_end: int,
        N: int,
        max_rank: int,
        correlation: float,
        dataset_dir: str,
        output_dir: str,
        seed: int,
        train_epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bench_dir = create_recursive_folder(output_dir, 'tt_cross_benchmark')
    logger = setup_logging(bench_dir, 'benchmark.log')
    logger.info('Starting TT-cross vs FNN TT benchmark')
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    for d in range(d_start, d_end + 1, 4):
        logger.info(f'===== Dimension d={d} =====')
        train_file, val_file, test_file, info = find_dataset(
            dataset_dir,
            d=d,
            N=N,
            correlation=correlation,
            format='TT',
            semi_supervised=True,
            max_rank=max_rank,
        )
        train_ds = TTDataset(torch.load(train_file))
        val_ds = TTDataset(torch.load(val_file))
        test_ds = TTDataset(torch.load(test_file))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        # Train model
        ranks = build_tt_ranks(d, max_rank)
        model = FNNNeuralMPS(ranks=ranks, n=N, input_size=info['input_size']).to(device)
        early = EarlyStopping(patience=10, delta=0.0, offset=5)
        hist = train_fnn_model(model, train_loader, val_loader, N, d, device, train_epochs, lr, weight_decay, logger, early)
        # Save trained model checkpoint for this dimension
        model_path = os.path.join(bench_dir, f'model_d{d}.pth')
        torch.save({
            'd': d,
            'N': N,
            'ranks': ranks,
            'input_size': info['input_size'],
            'state_dict': model.state_dict(),
            'train_history': hist['train'],
            'val_history': hist['val'],
        }, model_path)
        logger.info(f'[d={d}] Saved model checkpoint to {model_path}')

        # Helper to unpack parameters -> A, c, cov
        def unpack_params(params_tensor: torch.Tensor):
            # params shape: [param_dim]; structure: [A] + c (d) + upper-tri cov (d*(d+1)/2)
            p = params_tensor.detach().cpu().numpy()
            A_val = float(p[0])
            c_vec = p[1:1 + d]
            cov_ut = p[1 + d:]
            cov = np.zeros((d, d), dtype=np.float64)
            idxs = np.triu_indices(d)
            cov[idxs] = cov_ut
            # mirror to lower triangle
            cov = cov + np.triu(cov, 1).T
            return A_val, c_vec, cov

        # Per-sample evaluation (test set)
        nn_mse_list, nn_mae_list = [], []
        cross_mse_list, cross_mae_list = [], []
        model.eval()
        crit_mse = nn.MSELoss(reduction='mean')
        crit_mae = nn.L1Loss(reduction='mean')
        logger.info(f'[d={d}] Generating TT-cross approximations for {len(test_ds)} test samples')
        for idx in range(len(test_ds)):
            params, _ = test_ds[idx]  # ignore stored diagonal target; recompute analytic ground truth
            params_device = params.to(device, dtype=torch.float32).unsqueeze(0)  # [1, P]
            A_val, c_vec, cov_mat = unpack_params(params)
            # Ground truth diagonal analytic
            grid = np.linspace(-1, 1, N)
            x = np.stack([grid] * d)  # (d, N)
            gt_diag = gaussian_target_function(x, A_val, c_vec, cov_mat).float().to(device)  # [N]

            # NN prediction cores & diagonal
            with torch.no_grad():
                pred_cores = model(params_device)
                pred_diag = reconstruct_diagonal(pred_cores, N, d).squeeze(0)  # [N]

            # TT-cross for this sample
            # Build domain & ranks per sample
            domain = [torch.arange(N, device=device) for _ in range(d)]
            ranks_internal = [max_rank] * (d - 1)
            tt_tensor = tn.cross(
                function=lambda *ix, A=A_val, c=c_vec, cov_matrix=cov_mat: function_wrapper(
                    *ix, A=A, c=c, cov_matrix=cov_matrix, N=N, device=device
                ),
                domain=domain,
                eps=1e-7,
                ranks_tt=ranks_internal,
                max_iter=100,
                early_stopping_patience=5,
                early_stopping_tolerance=1e-8,
                verbose=False,
                suppress_warnings=True,
                device=device,
            )
            if isinstance(tt_tensor, tuple):
                tt_tensor = tt_tensor[0]
            cross_cores = [core.unsqueeze(0) for core in tt_tensor.cores]
            cross_diag = reconstruct_diagonal(cross_cores, N, d).squeeze(0)

            # Compute errors
            nn_mse_list.append(crit_mse(pred_diag, gt_diag).item())
            nn_mae_list.append(crit_mae(pred_diag, gt_diag).item())
            cross_mse_list.append(crit_mse(cross_diag, gt_diag).item())
            cross_mae_list.append(crit_mae(cross_diag, gt_diag).item())
            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_ds):
                logger.info(f'[d={d}] Processed {idx+1}/{len(test_ds)} test samples')

        # Aggregate metrics
        nn_mse = float(np.mean(nn_mse_list))
        nn_mae = float(np.mean(nn_mae_list))
        nn_rmse = math.sqrt(nn_mse)
        cross_mse = float(np.mean(cross_mse_list))
        cross_mae = float(np.mean(cross_mae_list))
        cross_rmse = math.sqrt(cross_mse)
        logger.info(f'[d={d}] NN Avg MSE={nn_mse:.6e} RMSE={nn_rmse:.6e} MAE={nn_mae:.6e}')
        logger.info(f'[d={d}] TT-Cross Avg MSE={cross_mse:.6e} RMSE={cross_rmse:.6e} MAE={cross_mae:.6e}')

        entry = {
            'd': d,
            'nn': {
                'mse': nn_mse,
                'rmse': nn_rmse,
                'mae': nn_mae,
                'train_hist': hist['train'],
                'val_hist': hist['val'],
                'per_sample_mse': nn_mse_list,
                'per_sample_mae': nn_mae_list,
            },
            'tt_cross': {
                'mse': cross_mse,
                'rmse': cross_rmse,
                'mae': cross_mae,
                'per_sample_mse': cross_mse_list,
                'per_sample_mae': cross_mae_list,
            },
        }
        results.append(entry)
        with open(os.path.join(bench_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    logger.info('Benchmark complete.')


def main():
    parser = argparse.ArgumentParser(description='Benchmark TT-cross vs FNN NeuralMPS across dimensions.')
    parser.add_argument('--d-start', type=int, default=10)
    parser.add_argument('--d-end', type=int, default=30)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--max-rank', type=int, default=15)
    parser.add_argument('--correlation', type=float, default=0.75)
    parser.add_argument('--dataset-dir', type=str, default='../data/datasets')
    parser.add_argument('--output-dir', type=str, default='../output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    args = parser.parse_args()
    benchmark(
        d_start=args.d_start,
        d_end=args.d_end,
        N=args.N,
        max_rank=args.max_rank,
        correlation=args.correlation,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == '__main__':
    main()

