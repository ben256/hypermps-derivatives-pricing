from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import tntorch as tn
from torch.utils.data import DataLoader

from data_processing.dataset import TTDataset
from model.neural_mps import NeuralMPS
from train_test.utils import setup_logging


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


def eval_tt2(tt_cores, idx):
    v = tt_cores[0][0, idx[0], :]
    for k in range(1, len(tt_cores)):
        Gk_slice = tt_cores[k][:, idx[k], :]
        v = v @ Gk_slice
    return v.squeeze()


def eval_tt(
        tt_cores: List[torch.Tensor],
        domain: List[torch.Tensor]
) -> torch.Tensor:
    B = tt_cores[0].shape[0]
    N = domain[0].shape[0]
    device = domain[0].device

    result = torch.zeros(B, N, device=device)

    for i in range(N):
        idx = [int(domain[k][i].item()) for k in range(len(domain))]

        G1 = tt_cores[0]
        v  = G1[:, 0, idx[0], :]

        for k in range(1, len(tt_cores)):
            Gk = tt_cores[k]
            slice_k = Gk[:, :, idx[k], :]
            v = torch.einsum('bi,bij->bj', v, slice_k)

        result[:, i] = v.view(B)

    return result


def function_wrapper(*ix, A, c, cov_matrix, N):
    d = len(ix)
    x_vector = []
    for i in range(d):
        indices = ix[i].cpu().numpy() if isinstance(ix[i], torch.Tensor) else ix[i]
        indices = indices.astype(int)
        x_vector.append(np.take(np.linspace(-1, 1, N), indices))
    out = target_function(np.stack(x_vector), A, c, cov_matrix)

    return out


def plot(
        d: int = 4,
        max_rank: int = 20,
        model_path: str = '../data/models/TT_d4_corr0-1_split',
        dataset_path: str = '../data/datasets/TT_d4_corr0-1',
        base: str = '',
        device: str = 'cpu',
):
    logger = setup_logging(save_to_file=False)

    torch.manual_seed(42)
    device = torch.device(device)
    logger.info(f"Using device: {device}{device.index if device.type == 'cuda' else ''}")

    # Load model
    model_data = torch.load(f'{model_path}/best_model.pth', map_location=device)
    model = NeuralMPS(
        ranks=[1, 20, 20, 20, 1],
        n=32,
        input_size=21,
        decoder_type='split',
    )
    model.to(device)
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    N = 32
    grid_1d = np.linspace(-1, 1, N)
    domain = [torch.arange(N, device=device) for _ in range(d)]

    A_range = [0.2, 1.0]
    c_range = [-0.5, 0.5]

    fig, axes = plt.subplots(4, 2, figsize=(12, 16), tight_layout=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        rng = np.random.default_rng()
        A = rng.uniform(low=A_range[0], high=A_range[1])
        c = rng.uniform(low=c_range[0], high=c_range[1], size=d)
        cov_matrix = generate_covariance_matrix(rng, d, correlation=0.1)

        params = np.concatenate([[A], c, cov_matrix.flatten()])
        params = torch.from_numpy(params).to(torch.float32).to(device)

        with torch.no_grad():
            tt = model.predict(params)
            tt_output = eval_tt(tt, domain)[0, :]

        target_function_output = function_wrapper(
            *domain,
            A=A,
            c=c,
            cov_matrix=cov_matrix,
            N=N
        )

        ax.plot(grid_1d, target_function_output.detach().numpy(), label='Target Function', color='blue', linewidth=2)
        ax.plot(grid_1d, tt_output.cpu().detach().numpy(), label='TT Output', color='orange', linestyle='--', linewidth=2)

        ax.set_title(f'Sample {i + 1}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(visible=True, which='major', linestyle='-', alpha=0.5)
        ax.grid(visible=True, which='minor', linestyle='--', alpha=0.2)
        ax.minorticks_on()
        ax.legend()

    plt.suptitle('Target Function vs. TT Representation (Split Decoder)', fontsize=16)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot()
