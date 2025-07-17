import numpy as np
import torch
import torch.nn as nn


from layers import ConvTransposeLayer, FCLayer


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

        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.n_i, self.r_ip1))  # to go from multiple of 2 to exact size

        self.final_conv = nn.Conv2d(
            in_channels,
            self.r_i,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
