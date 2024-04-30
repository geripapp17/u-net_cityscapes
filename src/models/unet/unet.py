import torch
from torch import nn
import torch.functional as F

from typing import Optional, Tuple, List


class DoubleConv(nn.Module):
    """"""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: Optional[int] = None) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    """Implementation of the Decoder class for U-Net architecture."""

    def __init__(self, in_channels: int = 3, features: Tuple[int] = (64, 128, 256, 512)) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = in_channels
        for feature in features:
            self.layers.append(DoubleConv(in_channels=in_ch, out_channels=feature))
            in_ch = feature

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = []

        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.max_pool(x)

        return x, skip_connections


class Decoder(nn.Module):
    """Implementation of the Decoder class for U-Net architecture."""

    def __init__(self, in_channels: int = 1024, features: Tuple[int] = (64, 128, 256, 512)) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        in_ch = in_channels
        for feature in reversed(features):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.layers.append(DoubleConv(in_channels=in_ch, out_channels=feature))

            in_ch = feature

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:

        it = iter(self.layers)
        for conv_tr in it:
            double_conv = next(it)

            x = conv_tr(x)
            x = torch.cat((skip_connections.pop(), x), dim=1)
            x = double_conv(x)

        return x


class UNet(nn.Module):
    """Implementation of the U-Net architecture."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Tuple[int] = (64, 128, 256, 512),
        bottleneck_channels: int = 1024,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, features=features)
        self.decoder = Decoder(in_channels=bottleneck_channels, features=features)
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=bottleneck_channels)
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = self.final_conv(x)

        return x
