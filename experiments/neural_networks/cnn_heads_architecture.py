import torch.nn as nn

from layers import FCLayer
from core_decoders import CoreDecoder


class HeadsDecoder(nn.Module):
    def __init__(
            self,
            latent_size: int,
            ranks: list[int],
            n: int,
            dropout: float,
    ):
        super().__init__()

        num_cores = len(ranks) - 1
        core_dims = [(ranks[i], n, ranks[i+1]) for i in range(len(ranks) - 1)]

        self.core_decoders = nn.ModuleList([
            CoreDecoder(latent_size, core_dims[i], dropout)
        for i in range(num_cores)])

    def forward(self, x):
        cores = [head(x) for head in self.core_decoders]
        return cores


class NeuralMPS(nn.Module):
    def __init__(
            self,
            ranks: list[int],
            hidden_size: int = 128,
            latent_size: int = 64,
            input_size: int = 15,

            n:int = 2,

            num_layers: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = FCLayer(input_size, hidden_size, dropout)
        self.hidden_layers = nn.ModuleList([
            FCLayer(hidden_size, hidden_size, dropout) for _ in range(num_layers - 1)
        ])
        self.fc2 = FCLayer(hidden_size, latent_size, dropout)  # maybe don't need this
        self.cnn = HeadsDecoder(latent_size, ranks, n, dropout)

    def forward(self, mu):
        """
        mu: [batch_size, input_size (15 atm)]
        """
        x = self.fc1(mu)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.fc2(x)
        cores = self.cnn(x)
        return cores
