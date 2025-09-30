import json
import logging
import math
import os
import sys
from glob import glob

import numpy as np
import torch

from distributions import mixture_logpdf, sample_mixture_params


def bits_to_idx_nd(bits: torch.Tensor, d: int, N: int) -> torch.Tensor:
    """
    bits: [B, S, K]
    returns idx_nd: [B, S, d]
    """
    B, S, K = bits.shape
    k = int(math.log2(N))
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


def flat_to_idx_nd(idx_flat: torch.Tensor, d: int, N: int) -> torch.Tensor:
    """
    idx_flat: [S] or [B,S]
    returns: [S,d] (if 1D input) or [B,S,d]
    """
    original_shape = idx_flat.shape
    if idx_flat.dim() == 1:
        idx_flat = idx_flat.view(1, -1)
    B, S = idx_flat.shape
    strides = torch.tensor([N ** (d - 1 - i) for i in range(d)],
                           device=idx_flat.device, dtype=torch.long)  # [d]
    coords = []
    remaining = idx_flat
    for i in range(d):
        stride = strides[i]
        coord = (remaining // stride) % N
        coords.append(coord)  # [B,S]
        remaining = remaining % stride
    idx_nd = torch.stack(coords, dim=-1)  # [B,S,d]
    if len(original_shape) == 1:
        return idx_nd.view(S, d)
    return idx_nd  # [B,S,d]


def idx_nd_to_bits(idx_nd: torch.Tensor, d: int, N: int) -> torch.Tensor:
    """
    idx_nd: [B, S, d]
    returns bits: [B, S, K] with K=d*log2(N), MSB-first per coordinate
    """
    B, S, d_ = idx_nd.shape
    assert d_ == d
    k = int(math.log2(N))
    x = idx_nd.to(torch.long).unsqueeze(-1)  # [B,S,d,1]
    bits_list = [((x >> shift) & 1) for shift in range(k - 1, -1, -1)]  # MSB->LSB
    b = torch.cat(bits_list, dim=-1)  # [B,S,d,k]
    return b.view(B, S, d * k)  # [B,S,K]


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


def setup_logging(log_dir='../logs', log_file='training.log', save_to_file=True):
    """
    Log to both console and file.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    if save_to_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


def create_recursive_folder(output_dir='../output', subfolder='training'):
    tuning_folders = glob(f'{output_dir}/{subfolder}_*')
    folder_num = [int(x.split('_')[-1]) for x in tuning_folders]
    if len(folder_num) > 0:
        count = max(folder_num) + 1
    else:
        count = 0
    folder_path = f'{output_dir}/{subfolder}_{count}/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path

    else:
        raise FileExistsError


def find_dataset(
        dataset_dir: str,
        **kwargs,
):
    selected = []
    datasets = glob(f'{dataset_dir}/dataset_*')
    logging.info(f'Checking for datasets in {os.path.abspath(dataset_dir)}')
    datasets.sort()
    for dataset_folder in datasets:
        with open(f'{dataset_folder}/info.json', 'r') as f:
            info = json.load(f)

        filtered = {k: v for k, v in info.items() if k in kwargs}
        if filtered == kwargs:
            selected.append(dataset_folder)

    if selected:
        dataset_folder = selected[-1]
        train_file = f'{dataset_folder}/train.pt'
        val_file = f'{dataset_folder}/val.pt'
        test_file = f'{dataset_folder}/test.pt'
        info = json.load(open(f'{dataset_folder}/info.json', 'r'))

        logging.info('Found dataset matching criteria')
        return train_file, val_file, test_file, info

    raise FileNotFoundError(f"No dataset found matching criteria: {kwargs}")
