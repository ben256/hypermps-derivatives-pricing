import logging
import os
import random
import sys
from glob import glob

import torch


def _gray_from_binary(b: torch.Tensor) -> torch.Tensor:
    """
    Encode non-negative integer tensor b into Gray code.
    b: integer tensor (any shape)
    returns: integer tensor (same shape)
    """
    b = b.long()
    return b ^ (b >> 1)


def _binary_from_gray(g: torch.Tensor, L: int) -> torch.Tensor:
    """
    Decode Gray code integer tensor g to binary given bit-width L.
    Uses iterative XOR-unfold.
    g: integer tensor (any shape)
    L: number of bits
    returns: integer tensor (same shape)
    """
    g = g.long()
    b = g.clone()
    # Iterate up to L-1 times, safe for all L >= 1
    for shift in range(1, L):
        b = b ^ (b >> shift)
    return b


def bits_to_idx(
        bits: torch.Tensor,
        d: int,
        L: int,
        gray: bool = False,
):
    batch_size, n_samples, K = bits.shape
    separated_bits = bits.view(batch_size, n_samples, d, L).long()

    powers_of_two = (2 ** torch.arange(L - 1, -1, -1, device=bits.device)).view(1, 1, 1, L).long()
    gray_or_binary = (separated_bits * powers_of_two).sum(dim=-1)  # [B, S, d]

    if gray:
        indices_long = _binary_from_gray(gray_or_binary, L)
    else:
        indices_long = gray_or_binary

    return indices_long.to(dtype=bits.dtype)


def idx_to_bits(
        indices: torch.Tensor,
        d: int,
        L: int,
        gray: bool = False,
):
    batch_size, n_samples, _ = indices.shape
    idx_long = indices.long()

    if gray:
        code = _gray_from_binary(idx_long)  # [B, S, d]
    else:
        code = idx_long

    bits_long = torch.zeros(batch_size, n_samples, d, L, device=indices.device, dtype=torch.long)
    for i in range(L):
        power = L - 1 - i
        bits_long[:, :, :, i] = (code >> power) & 1

    bits = bits_long.view(batch_size, n_samples, d * L).to(dtype=indices.dtype)
    return bits


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
        device_name: str = '',
):
    """Return the best available torch.device.

    If device_name is provided (e.g. 'cpu', 'cuda', 'mps'), use it. Otherwise
    prefer Apple Metal ('mps') when available, then CUDA, else CPU.
    """
    if device_name == '':
        dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if dev.type != 'mps':
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device_name)

    logging.info(f'Using device: {dev}')
    return dev


def create_recursive_folder(output_dir='../output', subfolder='training'):
    tuning_folders = glob(f'{output_dir}/{subfolder}_*')
    folder_num = [int(x.split('_')[-1]) for x in tuning_folders]
    if len(folder_num) > 0:
        count = max(folder_num) + 1
    else:
        count = 0
    folder_path = f'{output_dir}/{subfolder}_{count}'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path
    else:
        raise FileExistsError


def bs_density_conditioning_param_len(d):
    return 4 * d  # S0, r, sigma, T for each dimension
