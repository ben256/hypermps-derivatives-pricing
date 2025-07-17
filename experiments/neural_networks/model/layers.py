import torch.nn as nn


class FCLayer(nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
            self,
            channel_size: int,
            kernel_size: int,
            padding: int,
            dropout: float,
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(channel_size, channel_size, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvTransposeLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            padding: int,
    ):
        super().__init__()

        self.conv_t = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_t(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
