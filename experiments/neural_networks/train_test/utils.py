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


def create_training_folder(output_dir='../output'):
    tuning_folders = glob(f'{output_dir}/training_*')
    folder_num = [int(x.split('_')[-1]) for x in tuning_folders]
    if len(folder_num) > 0:
        count = max(folder_num) + 1
    else:
        count = 0
    folder_path = f'{output_dir}/training_{count}/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return folder_path

    else:
        raise FileExistsError
