import numpy as np
import torch.nn as nn

from model.layers import ConvLayer, FCLayer, ConvTransposeLayer


class CoreDecoder(nn.Module):
    def __init__(
            self,
            latent_size: int,
            core_dims: tuple,
            dropout: float,
    ):
        super().__init__()
        self.r_i, self.n_i, self.r_ip1 = core_dims
        self.latent_size = latent_size

        # self.fc = FCLayer(self.latent_size, self.latent_size, dropout=dropout)

        num_stages = int(np.ceil(np.log2(max(self.n_i, self.r_ip1))))
        in_channels = self.latent_size

        self.deconv_layers = nn.ModuleList()
        for i in range(num_stages):
            out_channels = max(in_channels // 2, self.r_i)

            self.deconv_layers.append(
                ConvTransposeLayer(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.n_i, self.r_ip1))  # to go from power of 2 to exact size

        self.final_conv = nn.Conv2d(
            in_channels,
            self.r_i,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        """
        x: [batch_size, latent_size (64 atm)]
        """
        # x = self.fc(x)  # [batch_size, latent_size]
        x = x.view(-1, self.latent_size, 1, 1)  # [batch_size, latent_size, 1, 1]

        for layer in self.deconv_layers:
            x = layer(x)

        x = self.adaptive_pool(x)  # [batch_size, x, n_i, r_ip1]
        x = self.final_conv(x)  # [batch_size, r_i, n_i, r_ip1]
        return x


class SplitDecoder(nn.Module):
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
        cores = [decoder(x) for decoder in self.core_decoders]
        return cores


class SharedDecoder(nn.Module):
    def __init__(
            self,
            latent_size: int,
            ranks: list[int],
            n: int,
            dropout: float,
            channel_size: int = 32,
    ):
        super().__init__()
        self.H = max(ranks)
        self.W = max(ranks)
        self.ranks = ranks
        self.n = n
        self.channel_size = channel_size

        self.projection = nn.Linear(latent_size, channel_size * self.H * self.W)

        self.cnn1 = ConvLayer(channel_size, kernel_size=3, padding=1, dropout=dropout)
        self.cnn2 = ConvLayer(channel_size, kernel_size=3, padding=1, dropout=dropout)

        self.final_conv = nn.Conv2d(channel_size, self.n, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.projection(x)
        x = x.view(-1, self.channel_size, self.H, self.W)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.final_conv(x)  # [batch, n, H, W]

        cores = []
        for i in range(len(self.ranks) - 1):
            r_i = self.ranks[i]
            r_ip1 = self.ranks[i+1]

            core_i = x[:, :, :r_i, :r_ip1]  # [batch, n, r_i, r_{i+1}]
            core_i = core_i.permute(0, 2, 1, 3)  # [batch, r_i, n, r_{i+1}]
            cores.append(core_i)

        # channel_start = 0
        # num_cores = len(self.ranks) - 1
        # cores = []
        # for i in range(num_cores):
        #     ri, rip1 = self.ranks[i], self.ranks[i+1]
        #     xi = x[:, channel_start:channel_start+2, :ri, :rip1]
        #     channel_start += 2
        #     cores.append(xi.permute(0, 2, 1, 3))

        return cores


class NeuralMPS(nn.Module):
    def __init__(
            self,
            ranks: list[int],
            n: int,
            hidden_size: int = 128,
            latent_size: int = 64,
            input_size: int = 21,
            channel_size: int = 32,

            num_layers: int = 2,
            dropout: float = 0.1,

            decoder_type: str = 'shared'
    ):
        super().__init__()
        self.fc1 = FCLayer(input_size, hidden_size, dropout)
        self.hidden_layers = nn.ModuleList([
            FCLayer(hidden_size, hidden_size, dropout) for _ in range(num_layers - 1)
        ])
        self.fc2 = FCLayer(hidden_size, latent_size, dropout)  # maybe don't need this

        if decoder_type == 'shared':
            self.cnn = SharedDecoder(latent_size, ranks, n, dropout, channel_size)
        elif decoder_type == 'split':
            self.cnn = SplitDecoder(latent_size, ranks, n, dropout)
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

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
