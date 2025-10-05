import logging
import os
import random
import sys
from glob import glob
from typing import Optional

import torch


def bits_to_idx(
        bits: torch.Tensor,
        d: int,
        L: int,
):
    """
    Takes binary bits and returns the n-dimensional indices.
    """
    batch_size, n_samples, K = bits.shape
    separated_bits = bits.view(batch_size, n_samples, d, L)
    powers_of_two = (2 ** torch.arange(L - 1, -1, -1, device=bits.device)).view(1, 1, 1, L)
    indices = (separated_bits * powers_of_two).sum(dim=-1)
    return indices


def idx_to_bits(
        indices: torch.Tensor,
        d: int,
        L: int,
):
    """
    Takes n-dimensional indices and returns the binary bits.
    """
    batch_size, n_samples, _ = indices.shape
    bits = torch.zeros(batch_size, n_samples, d, L, device=indices.device)

    for i in range(L):
        power = L - 1 - i
        bits[:, :, :, i] = (indices // (2 ** power)) % 2

    bits = bits.view(batch_size, n_samples, d * L)
    return bits


def generate_covariance_matrix(
        rng: torch.Generator,
        d: int,
        correlation: Optional[float] = 0.5
):

    stds = torch.empty(d).uniform_(0.6, 1.4, generator=rng)
    corr_matrix = torch.full((d, d), correlation)
    corr_matrix.fill_diagonal_(1.0)

    try:
        torch.linalg.cholesky(corr_matrix)
    except RuntimeError:
        eigenvalues, eigenvectors = torch.linalg.eigh(corr_matrix)
        eigenvalues[eigenvalues < 0] = 0
        corr_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T

    S = torch.diag(stds)
    cov_matrix = S @ corr_matrix @ S
    return cov_matrix


def gaussian_conditioning_param_len(
        d: int,
):
    return d + (d * (d + 1)) // 2


def seed_everything(
        seed: int = 42
):
    """
    SEED EVERYTHING WHAT IS THE ANSWER TO LIFE THE UNIVERSE AND EVERYTHING
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


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


def setup_device(
        device: str = '',
):
    if device == '':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if device.type != 'mps':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    logging.info(f'Using device: {device}')
    return device



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
