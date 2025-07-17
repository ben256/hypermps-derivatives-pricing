import torch.nn as nn

from layers import ConvLayer, FCLayer


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
            hidden_size: int = 128,
            output_size: int = 64,
            input_size: int = 15,

            channel_size: int = 32,
            kernel_size: int = 3,
            padding: int = 1,

            num_layers: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = FCLayer(input_size, hidden_size, dropout)
        self.hidden_layers = nn.ModuleList([
            FCLayer(hidden_size, hidden_size, dropout) for _ in range(num_layers - 1)
        ])
        self.fc2 = FCLayer(hidden_size, output_size, dropout)  # maybe don't need this
        self.cnn = SharedDecoder(output_size, channel_size, kernel_size, padding, ranks, dropout)

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


