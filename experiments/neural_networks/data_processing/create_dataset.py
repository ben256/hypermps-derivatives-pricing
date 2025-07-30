import argparse
import os

import numpy as np
import tntorch as tn
import torch
from tqdm import tqdm


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
        cov_matrix: np.ndarray
):
    c = c.reshape(-1, 1)
    cov_inv = np.linalg.inv(cov_matrix)

    diff = x - c
    exponent_term = -0.5 * np.einsum('ib,ij,jb->b', diff, cov_inv, diff)

    return torch.from_numpy(A * np.exp(exponent_term))


def function_wrapper(*ix, A, c, cov_matrix, N, type, device):
    if type == 'TT':
        d = len(ix)
        x_vector = []
        for i in range(d):
            indices = ix[i].cpu().numpy() if isinstance(ix[i], torch.Tensor) else ix[i]
            indices = indices.astype(int)
            x_vector.append(np.take(np.linspace(-1, 1, N), indices))
        out = target_function(np.stack(x_vector), A, c, cov_matrix)

    elif type == 'QTT':
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
        max_rank: int,
        dataset_path: str,
        correlation: float,
        format: str,
        device: str,
):
    device = torch.device(device)
    initial_seed = 42
    A_range = [0.2, 1.0]
    c_range = [-0.5, 0.5]

    if format == 'TT':
        N = 32
        domain = [torch.arange(N, device=device) for _ in range(d)]
        ranks = [max_rank] * (d - 1)

    elif format == 'QTT':
        N = 32
        k = int(np.log2(N))
        domain = [torch.arange(2, device=device) for _ in range(d * k)]
        ranks = [max_rank] * (d * k - 1)

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
        cov_triu = np.triu(cov_matrix).flatten()
        cov_triu_input = cov_triu[cov_triu != 0.0]

        tt_tensor= tn.cross(
            function=lambda *ix: function_wrapper(*ix, A=A, c=c, cov_matrix=cov_matrix, N=N, type=format, device=device),
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

        params = np.concatenate([[A], c, cov_triu.flatten()])
        params = torch.from_numpy(params).to(torch.float32).to(device)

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

    dataset_folder = f'{format}_d{d}_corr{str(correlation).replace(".", "-")}'
    os.makedirs(f'{dataset_path}/{dataset_folder}', exist_ok=True)

    torch.save(train_data, f'{dataset_path}/{dataset_folder}/train.pt')
    torch.save(val_data, f'{dataset_path}/{dataset_folder}/val.pt')
    torch.save(test_data, f'{dataset_path}/{dataset_folder}/test.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n-samples', type=int, default=100000)
    parser.add_argument('--d', type=int, default=4)
    parser.add_argument('--max-rank', type=int, default=20)
    # parser.add_argument('--dataset-path', type=str, default='/cs/student/projects3/cf/2024/bnaylor/dev/hypermps-derivatives-pricing/experiments/neural_networks/data/datasets')
    parser.add_argument('--dataset-path', type=str, default='../data/datasets')
    parser.add_argument('--correlation', type=float, default=0.3)
    parser.add_argument('--format', type=str, choices=['TT', 'QTT'], default='TT')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    create_datasets(
        n_samples=args.n_samples,
        d=args.d,
        max_rank=args.max_rank,
        dataset_path=args.dataset_path,
        correlation=args.correlation,
        format=args.format,
        device=args.device,
    )