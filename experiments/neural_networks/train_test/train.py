import json
import os
import copy
import argparse

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from data_processing.dataset import TTDataset
from model.neural_mps import NeuralMPS
from train_test.utils import create_training_folder, setup_logging


class EarlyStopping:
    def __init__(
            self,
            patience: int = 5,
            delta: float = 0.0,
            offset: int = 5,
            verbose: bool = False,
            logger = None,
    ):
        self.patience = patience
        self.delta = delta
        self.offset = offset
        self.verbose = verbose
        self.logger = logger
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model, epoch=None):
        if epoch >= self.offset:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_model_state = copy.deepcopy(model.state_dict())
                if self.verbose:
                    self.logger.info(f"Initial validation loss: {val_loss:.6f}. Saving model.")
            elif val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                self.best_model_state = copy.deepcopy(model.state_dict())
                self.counter = 0
                if self.verbose:
                    self.logger.info(f"Validation loss improved to {val_loss:.6f}. Saving model.")
            else:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True


def train(
        batch_size: int = 200,
        learning_rate: int = 5e-6,
        num_training_epochs: int = 20,
        early_stopping_patience: int = 5,
        early_stopping_delta: float = 0.0,
        early_stopping_offset: int = 5,
        weight_decay: float = 0.01,
        dropout: float = 0.1,
        decoder_type: str = 'shared',
        dataset_dir: str = '../data/datasets',
        output_dir: str = '../output',
):
    output_dir = create_training_folder(output_dir)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    logger = setup_logging(output_dir)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}{device.index if device.type == 'cuda' else ''}")

    train_dataset = TTDataset(torch.load(f'{dataset_dir}/train.pt'))
    val_dataset = TTDataset(torch.load(f'{dataset_dir}/val.pt'))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    sample = train_dataset[0]
    dataset_ranks = [core.shape[0] for core in sample[1]] + [1]
    n = sample[1][0].shape[1]

    model = NeuralMPS(
        ranks=dataset_ranks,
        n=n,
        input_size=sample[0].shape[0],
        decoder_type=decoder_type,
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epoch = 0
    train_loss_history, validation_loss_history = [], []
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        delta=early_stopping_delta,
        offset=early_stopping_offset,
        verbose=True,
        logger=logger,
    )

    logger.info("Starting training loop")
    logger.info(f"Number of training epochs: {num_training_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Early stopping: Patience: {early_stopping_patience}, Delta: {early_stopping_delta}, Offset: {early_stopping_offset}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Dropout: {dropout}")
    logger.info(f"Decoder type: {decoder_type}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Output directory: {output_dir}")

    try:
        for epoch in range(num_training_epochs):
            model.train()
            epoch_train_loss = 0.0
            for batch_idx, (mu, tt_cores) in enumerate(train_dataloader):
                mu = mu.to(device, dtype=torch.float32)
                tt_cores = [core.to(device, dtype=torch.float32) for core in tt_cores]

                optimiser.zero_grad()
                output = model(mu)
                loss = 0.0
                for pred, true in zip(output, tt_cores):
                    loss += criterion(pred, true)
                loss = loss / len(output)

                loss.backward()
                optimiser.step()

                epoch_train_loss += loss.item()

                if batch_idx % 80 == 0:
                    progress = 100. * (batch_idx + 1) / len(train_dataloader)
                    logger.info(
                        f"Train Epoch: {epoch} [{batch_idx + 1}/{len(train_dataloader)} batches ({progress:.2f}%)]\tLoss: {loss.item():.8f}"
                    )

            avg_train_loss = epoch_train_loss / len(train_dataloader)
            train_loss_history.append(avg_train_loss)

            # Validation
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (mu, tt_cores) in enumerate(validation_dataloader):
                    mu = mu.to(device)
                    tt_cores = [core.to(device) for core in tt_cores]

                    output = model(mu)
                    loss = 0.0
                    for pred, true in zip(output, tt_cores):
                        loss += criterion(pred, true)
                    loss = loss / len(output)
                    epoch_val_loss += loss.item()

            avg_val_loss = epoch_val_loss / len(validation_dataloader)
            validation_loss_history.append(avg_val_loss)
            logger.info(f"Validation: Average loss: {avg_val_loss:.8f}")

            with open(f'{output_dir}/loss.json', 'w') as f:
                json.dump({'train': train_loss_history, 'validation': validation_loss_history}, f)

            # Check early stopping
            early_stopping(avg_val_loss, model, epoch)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered. Exiting training loop.")
                break

    except Exception as e:
        # Save checkpoint if an exception occurs
        error_checkpoint_path = os.path.join(checkpoint_dir, f"error_checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'train_loss_history': train_loss_history,
            'validation_loss_history': validation_loss_history,
        }, error_checkpoint_path)
        logger.error(f"Error encountered. Saved checkpoint: {error_checkpoint_path}")
        raise e

    # Load best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    final_model_path = os.path.join(output_dir, 'best_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history,
    }, final_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--num-training-epochs', type=int, default=100)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--early-stopping-delta', type=float, default=0.0001)
    parser.add_argument('--early-stopping-offset', type=int, default=10)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--decoder-type', type=str, choices=['shared', 'split'], default='split')
    parser.add_argument('--dataset-dir', type=str, default='../data/datasets/TT_d4_corr0-3')
    parser.add_argument('--output-dir', type=str, default='../output')

    args = parser.parse_args()

    train(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_training_epochs=args.num_training_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_delta=args.early_stopping_delta,
        early_stopping_offset=args.early_stopping_offset,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        decoder_type=args.decoder_type,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
    )