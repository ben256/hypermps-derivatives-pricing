import json
from glob import glob

import logging
import os
import sys


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
