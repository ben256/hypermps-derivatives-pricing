import numpy as np
import torch.nn as nn

from model.layers import ConvLayer, FCLayer, ConvTransposeLayer


class CoreDecoder(nn.Module):
    def __init__(
            self,
            latent_size: int,
            core_dims: tuple,
            dropout: float,
            feature_map_size: int = 64,
    ):
        super().__init__()
        self.r_i, self.n_i, self.r_ip1 = core_dims
        self.feature_map_size = feature_map_size

        self.fc = FCLayer(latent_size, self.feature_map_size, dropout=dropout)

        num_stages = int(np.ceil(np.log2(max(self.n_i, self.r_ip1))))
        in_channels = self.feature_map_size

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
        x = self.fc(x)  # [batch_size, feature_map_size]
        x = x.view(-1, self.feature_map_size, 1, 1)  # [batch_size, feature_map_size, 1, 1]

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
            channel_size: int,
            kernel_size: int,
            padding: int,
            ranks: list[int],
            dropout: float,
    ):
        super().__init__()
        self.H = max(ranks)
        self.W = max(ranks)
        self.channel_size = channel_size
        self.ranks = ranks

        self.projection = nn.Linear(latent_size, channel_size * self.H * self.W)

        self.cnn1 = ConvLayer(channel_size, kernel_size, padding, dropout)
        self.cnn2 = ConvLayer(channel_size, kernel_size, padding, dropout)

        self.heads = nn.ModuleList([
            nn.Conv2d(channel_size, i, kernel_size=kernel_size, padding=padding) for i in ranks[:-1]
        ])

    def forward(self, x):
        """
        x: [batch_size, latent_size (64 atm)]
        """
        x = self.projection(x)
        x = x.view(-1, self.channel_size, self.H, self.W)
        x = self.cnn1(x)
        x = self.cnn2(x)

        cores = []
        for i, head in enumerate(self.heads):
            r_i = self.ranks[i]
            r_ip1 = self.ranks[i+1]
            x_slice = x[:, :, :r_i, :r_ip1]  # [batch_size, channel_size, r_i, r_{i+1}]
            core_i = head(x_slice)  # [batch_size, n_i, r_i, r_{i+1}]
            core_i = core_i.permute(0, 2, 1, 3)  # [batch_size, r_i, n_i, r_{i+1}]
            cores.append(core_i)

        # this is probably a bit simpler, maybe should have just done this to start...
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
            n: int = 2,
            hidden_size: int = 128,
            latent_size: int = 64,
            input_size: int = 15,

            channel_size: int = 32,
            kernel_size: int = 3,
            padding: int = 1,

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

        if decoder_type == 'split':
            self.cnn = SharedDecoder(latent_size, channel_size, kernel_size, padding, ranks, dropout)
        else:
            self.cnn = SplitDecoder(latent_size, ranks, n, dropout)

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

