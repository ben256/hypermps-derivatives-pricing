import argparse
import json
import numpy as np
import torch
from tqdm import tqdm

from scipy.special import logsumexp
from utils import create_recursive_folder
from distributions import (
    generate_covariance_matrix,
    gaussian_logpdf,
    mixture_logpdf,
    sample_mixture_params,
)


RUN_TTCROSS = True


def renormalize_logpdf_on_grid(
        logp: np.ndarray,
        dx: float, d: int
) -> np.ndarray:
    logZ = logsumexp(logp) + d * np.log(dx)
    return logp - logZ


def create_datasets(
        n_samples: int,
        d: int,
        N: int,
        dataset_path: str,
        correlation: float,
        initial_seed: int,
        device: str,
        distribution: str,
        n_components: int,
        grid_min: float,
        grid_max: float,
        output_space: str,
        renormalize_on_grid: bool,
):
    if output_space not in {"log", "density"}:
        raise ValueError("output_space must be 'log' or 'density'.")
    if distribution == "mixture1d" and d != 1:
        raise ValueError("mixture1d requires d=1.")
    if distribution == "mixture2d" and d != 2:
        raise ValueError("mixture2d requires d=2.")

    device = torch.device(device)

    k = int(round(np.log2(N)))
    if 2 ** k != N:
        raise ValueError("N must be a power of 2.")

    grid = np.linspace(grid_min, grid_max, N, dtype=np.float64)
    dx = float((grid_max - grid_min) / (N - 1))

    train_size, val_size, test_size = 0.8, 0.1, 0.1
    data = []

    for n in tqdm(range(n_samples)):
        seed = initial_seed + n
        rng = np.random.default_rng(seed)

        if distribution == "gaussian":
            c = rng.uniform(-0.5, 0.5, size=d)
            cov = generate_covariance_matrix(rng, d, correlation=correlation)
            params = np.concatenate((c, np.triu(cov)[np.triu_indices(d)]))

            if d == 1:
                x = grid.reshape(1, -1)
            elif d == 2:
                xx, yy = np.meshgrid(grid, grid, indexing="ij")
                x = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=0)
            else:
                raise ValueError("Only for up to 2D.")

            cov_inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            log_norm = -0.5 * (d * np.log(2 * np.pi) + logdet)
            logp = gaussian_logpdf(x, c, cov_inv, log_norm)

        elif distribution in ("mixture1d", "mixture2d"):
            weights, means, covs, invs, log_norms = sample_mixture_params(rng, d, n_components=n_components)
            params_list = [weights.flatten(), means.flatten()]
            for cov in covs:
                params_list.append(np.triu(cov)[np.triu_indices(d)])
            params = np.concatenate(params_list)

            if d == 1:
                x = grid.reshape(1, -1)
            elif d == 2:
                xx, yy = np.meshgrid(grid, grid, indexing="ij")
                x = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=0)
            else:
                raise ValueError("Only for up to 2D! BIG NO!")

            logp = mixture_logpdf(x, weights, means, invs, log_norms)

        else:
            raise ValueError(f"Unsupported distribution {distribution}")

        if RUN_TTCROSS:
            try:
                import tntorch as tn
            except ImportError:
                print("[TT-CROSS] tntorch not installed; skipping.")
            else:
                # Build bit-to-grid mapping
                k = int(round(np.log2(N)))
                assert 2 ** k == N, "N must be a power of 2."
                D = d * k
                grid_t = torch.linspace(grid_min, grid_max, N, device=device)
                bitw = (2 ** torch.arange(k - 1, -1, -1, device=device)).long()

                def map_bits_to_x(*ix):
                    B = ix[0].shape[0]
                    bits = torch.stack(ix, dim=0).view(d, k, B).long()
                    idx = (bits * bitw.view(1, k, 1)).sum(dim=1)
                    return grid_t.index_select(0, idx.view(-1)).view(d, B)

                # Analytic target (density space)
                if distribution == "gaussian":
                    mu = c.astype(np.float64)
                    cov_np = cov.astype(np.float64)
                    cov_inv = np.linalg.inv(cov_np)
                    _, logdet = np.linalg.slogdet(cov_np)
                    log_norm = -0.5 * (d * np.log(2 * np.pi) + logdet)

                    def f(*ix):
                        x = map_bits_to_x(*ix)
                        x_np = x.detach().cpu().numpy()
                        logp_loc = gaussian_logpdf(x_np, mu, cov_inv, log_norm)
                        return torch.from_numpy(np.exp(logp_loc)).to(device=device, dtype=torch.float32)

                elif distribution in ("mixture1d", "mixture2d"):
                    w = weights
                    m = means
                    invs_loc = invs
                    log_norms_loc = log_norms

                    def f(*ix):
                        x = map_bits_to_x(*ix)
                        x_np = x.detach().cpu().numpy()
                        logp_loc = mixture_logpdf(x_np, w, m, invs_loc, log_norms_loc)
                        return torch.from_numpy(np.exp(logp_loc)).to(device=device, dtype=torch.float32)
                else:
                    raise ValueError(f"[TT-CROSS] Unsupported distribution {distribution}")

                domain = [torch.arange(2, device=device) for _ in range(D)]
                max_rank_tt = 15
                ranks_tt = [min(max_rank_tt, 1 << min(i + 1, D - (i + 1))) for i in range(D - 1)]

                # Run TT-cross
                tt_tensor = tn.cross(
                    function=f,
                    domain=domain,
                    eps=1e-7,
                    ranks_tt=ranks_tt,
                    max_iter=100,
                    early_stopping_patience=3,
                    early_stopping_tolerance=1e-8,
                    device=device,
                )

                # Save TT cores for reference
                tt_folder = create_recursive_folder(dataset_path, 'tt_refs')
                cores_cpu = [core.detach().cpu().to(torch.float32) for core in tt_tensor.cores]
                torch.save(
                    {
                        "cores": cores_cpu,
                        "ranks": tt_tensor.ranks,
                        "N": N,
                        "d": d,
                        "distribution": distribution,
                        "sample_index": n,
                        "seed": seed,
                    },
                    f"{tt_folder}/cores_{distribution}_d{d}_N{N}_idx{n:06d}.pt",
                )

        params_t = torch.from_numpy(params).to(device)

        if renormalize_on_grid:
            logp = renormalize_logpdf_on_grid(logp, dx, d)
        target = torch.from_numpy(logp.astype(np.float32))
        if output_space == "density":
            target = torch.exp(target)

        data.append((params_t, target.to(device)))

    rng = np.random.default_rng(initial_seed)
    rng.shuffle(data)

    train_end = int(train_size * len(data))
    val_end = train_end + int(val_size * len(data))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    dataset_folder = create_recursive_folder(dataset_path, 'dataset')
    torch.save(train_data, f'{dataset_folder}/train.pt')
    torch.save(val_data, f'{dataset_folder}/val.pt')
    torch.save(test_data, f'{dataset_folder}/test.pt')

    dataset_info = {
        'd': d,
        'n_samples': n_samples,
        'correlation': correlation,
        'N': N,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'initial_seed': initial_seed,
        'distribution': distribution,
        'n_components': n_components,
        'input_size': len(params),
        'grid_min': grid_min,
        'grid_max': grid_max,
        'output_space': output_space,
        'renormalize_on_grid': renormalize_on_grid,
    }
    with open(f'{dataset_folder}/info.json', 'w') as f:
        json.dump(dataset_info, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--d', type=int, default=5)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--dataset-path', type=str, default='./datasets')
    parser.add_argument('--correlation', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--distribution', type=str, choices=['gaussian', 'mixture1d', 'mixture2d'], default='mixture1d')
    parser.add_argument('--n-components', type=int, default=1)
    parser.add_argument('--grid-min', type=float, default=-4.0)
    parser.add_argument('--grid-max', type=float, default=4.0)
    parser.add_argument('--output-space', type=str, choices=['log', 'density'], default='log')
    parser.add_argument('--renormalize-on-grid', action='store_true', default=False)

    args = parser.parse_args()
    create_datasets(
        n_samples=args.n_samples,
        d=args.d,
        N=args.N,
        dataset_path=args.dataset_path,
        correlation=args.correlation,
        initial_seed=args.seed,
        device=args.device,
        distribution=args.distribution,
        n_components=args.n_components,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        output_space=args.output_space,
        renormalize_on_grid=args.renormalize_on_grid,
    )
