import argparse
import json

import numpy as np
import tntorch as tn
import torch
from tqdm import tqdm

from train.utils import create_recursive_folder


def generate_covariance_matrix(
        rng: np.random.Generator,
        d: int,
        correlation: float = None,
):
    stds = rng.uniform(0.1, 1.0, size=d)
    corr_matrix = np.full((d, d), correlation if correlation is not None else 0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    try:
        np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues[eigenvalues < 0] = 0
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    S = np.diag(stds)
    cov_matrix = S @ corr_matrix @ S
    return cov_matrix


def target_function(
        x: np.ndarray,
        A: float,
        c: np.ndarray,
        cov_matrix: np.ndarray,
):
    c = c.reshape(-1, 1)
    cov_inv = np.linalg.inv(cov_matrix)
    diff = x - c  # shape: (d, b)
    quad = np.einsum('ib,ij,jb->b', diff, cov_inv, diff)
    y_log10 = np.log10(A) - 0.5 * quad / np.log(10.0)
    return torch.from_numpy(y_log10)


def function_wrapper(*ix, A, c, cov_matrix, N, type, device):
    if type == 'TT':
        d = len(ix)
        x_vector = []
        for i in range(d):
            indices = ix[i].cpu().numpy() if isinstance(ix[i], torch.Tensor) else ix[i]
            indices = indices.astype(int)
            x_vector.append(np.take(np.linspace(-1, 1, N), indices))
        out = target_function(np.stack(x_vector), A, c, cov_matrix)

    elif type == 'QTT':  # Binary base
        d = len(c)
        k = int(np.log2(N))
        x_vector = []
        for i in range(d):
            bits = ix[i * k : (i + 1) * k]
            bits_arr = [
                b.cpu().numpy().astype(int) if isinstance(b, torch.Tensor)
                else np.array(b, dtype=int)
                for b in bits
            ]
            idx = np.zeros_like(bits_arr[0], dtype=int)
            for j, bit in enumerate(bits_arr):
                idx += bit << (k - 1 - j)
            x_vector.append(np.linspace(-1, 1, N)[idx])
        out = target_function(np.stack(x_vector), A, c, cov_matrix)

    else:
        raise ValueError(f"Unsupported type: {type}")

    return out.to(device)


def create_datasets(
        n_samples: int,
        d: int,
        N: int,
        max_rank: int,
        dataset_path: str,
        correlation: float,
        format: str,
        initial_seed: int,
        semi_supervised: bool,
        device: str,
        standardize_targets: bool = False,
):
    device = torch.device(device)
    A_range = [0.2, 1.0]
    c_range = [-0.5, 0.5]
    grid = np.linspace(-1, 1, N)

    if format == 'TT':
        domain = [torch.arange(N, device=device) for _ in range(d)]
        ranks = [1] + [max_rank] * (d - 1) + [1]

    elif format == 'QTT':
        k = int(round(np.log2(N)))
        D = d * k
        domain = [torch.arange(2, device=device) for _ in range(D)]

        ranks = [1]
        for i in range(1, D):
            r_i = 1 << min(i, D - i)          # 2 ** min(i, D - i)
            ranks.append(min(r_i, max_rank))
        ranks.append(1)

    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats are 'TT' and 'QTT'.")

    train_size, val_size, test_size = 0.8, 0.1, 0.1

    data = []

    for n in tqdm(range(n_samples)):
        seed = initial_seed + n
        rng = np.random.default_rng(seed)
        A = rng.uniform(low=A_range[0], high=A_range[1])
        c = rng.uniform(low=c_range[0], high=c_range[1], size=d)
        cov_matrix = generate_covariance_matrix(rng, d, correlation=correlation)
        cov_triu = np.triu(cov_matrix)
        cov_triu_input = cov_triu[np.triu_indices(d)]

        params_np = np.concatenate((np.array([A], dtype=np.float64), c, cov_triu_input))
        params = torch.from_numpy(params_np.astype(np.float32)).to(device)

        if semi_supervised:
            target_function_output = target_function(
                grid,
                A=A,
                c=c,
                cov_matrix=cov_matrix
            )
            data.append((
                params,
                target_function_output.to(device)
            ))

        else:
            tt_tensor= tn.cross(
                function=lambda *ix: function_wrapper(
                    *ix,
                    A=A,
                    c=c,
                    cov_matrix=cov_matrix,
                    N=N,
                    type=format,
                    device=device
                ),
                domain=domain,
                eps=1e-7,
                ranks_tt=ranks[1:-1],
                max_iter=100,
                early_stopping_patience=3,
                early_stopping_tolerance=1e-8,
                verbose=False,
                suppress_warnings=True,
                device=device,
            )

            data.append((
                params,
                tt_tensor.cores,
            ))

    np.random.seed(initial_seed)
    np.random.shuffle(data)

    train_end = int(train_size * len(data))
    val_end = train_end + int(val_size * len(data))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 4) Standardize only when we have raw function values (semi-supervised)
    target_mean, target_std = None, None
    if semi_supervised and standardize_targets:
        # Flatten all targets in the training split to compute global μ/σ
        train_targets = torch.cat([t.reshape(-1) for _, t in train_data])
        target_mean = float(train_targets.mean().item())
        # Use population std; clamp to avoid divide-by-zero
        target_std = float(train_targets.std(unbiased=False).clamp_min(1e-8).item())

        def zscore(t: torch.Tensor) -> torch.Tensor:
            return (t - target_mean) / target_std

        train_data = [(p, zscore(t)) for p, t in train_data]
        val_data = [(p, zscore(t)) for p, t in val_data]
        test_data = [(p, zscore(t)) for p, t in test_data]

    dataset_folder = create_recursive_folder(dataset_path, 'dataset')

    torch.save(train_data, f'{dataset_folder}/train.pt')
    torch.save(val_data, f'{dataset_folder}/val.pt')
    torch.save(test_data, f'{dataset_folder}/test.pt')

    dataset_info = {
        'format': format,
        'd': d,
        'max_rank': max_rank,
        'n_samples': n_samples,
        'correlation': correlation,
        'N': N,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'initial_seed': initial_seed,
        'semi_supervised': semi_supervised,
        'input_size': d + (d * (d + 1)) // 2 + 1,
        'target_transform_type': 'zscore_log10' if (semi_supervised and standardize_targets) else 'log10_only',
        'target_mean': target_mean,
        'target_std': target_std,
    }
    with open(f'{dataset_folder}/info.json', 'w') as f:
        json.dump(dataset_info, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-samples', type=int, default=100000)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--max-rank', type=int, default=10)
    parser.add_argument('--dataset-path', type=str, default='../data/datasets')
    parser.add_argument('--correlation', type=float, default=0.5)
    parser.add_argument('--format', type=str, choices=['TT', 'QTT'], default='QTT')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--semi-supervised', type=bool, default=True)
    parser.add_argument('--standardize-targets', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    create_datasets(
        n_samples=args.n_samples,
        d=args.d,
        N=args.N,
        max_rank=args.max_rank,
        dataset_path=args.dataset_path,
        correlation=args.correlation,
        format=args.format,
        initial_seed=args.seed,
        semi_supervised=args.semi_supervised,
        device=args.device,
        standardize_targets=args.standardize_targets,
    )
