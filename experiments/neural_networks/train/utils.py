import json
from dataclasses import dataclass
from glob import glob

import logging
import os
import sys
from typing import Optional, Dict

import numpy as np
import torch
from torch import nn


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


def eval_tt(tt_cores, x_indices):
    v = tt_cores[0][:, :, x_indices[0], :]
    for i in range(1, len(tt_cores)):
        v = v @ tt_cores[i][:, :, x_indices[i], :]
    return v.sum()  # this maybe should be squeeze but haven't had time to test yet.


def eval_qtt(btt_cores, x_indices, N):
    k = int(np.log2(N))
    binary_indices = []
    for idx in x_indices:
        bits = bin(idx)[2:].zfill(k)
        binary_indices.extend([int(b) for b in bits])

    v = btt_cores[0][:, :, binary_indices[0], :]
    for i in range(1, len(btt_cores)):
        v = v @ btt_cores[i][:, :, binary_indices[i], :]
    return v.squeeze()


def build_ranks(
        format: str,
        d: int,
        N: int,
        max_rank: int,
        device: torch.device
):
    if format == 'TT':
        domain = [torch.arange(N, device=device) for _ in range(d)]
        ranks = [1] + [max_rank] * (d - 1) + [1]
        n_model = N
    elif format == 'QTT':
        k = int(np.log2(N))
        n_model = 2
        domain = [torch.arange(2, device=device) for _ in range(d * k)]

        ranks = [1] * (d * k + 1)
        for i in range(1, d * k):
            growth = 2 ** min(i, d * k - i)
            ranks[i] = min(growth, max_rank)

    else:
        raise ValueError(f"Unsupported format: {format}")

    return ranks, domain, n_model