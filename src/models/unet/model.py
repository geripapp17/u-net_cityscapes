import torch
import torch.nn as nn
from typing import List, Tuple


class Unet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features: List[int] = [64, 128, 256, 512]) -> None:
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels, features=features)
        self.decoder = Decoder(features=reversed(features))
        self.bottleneck = ConvBlock(in_channels=features[-1], out_channels=features[-1] * 2)
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = self.final_conv(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, features: List[int] = [64, 128, 256, 512]) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(ConvBlock(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_connections = list()
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.max_pool(x)

        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, features: List[int] = [512, 256, 128, 64]) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(
                nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2)
            )
            self.layers.append(ConvBlock(in_channels=feature * 2, out_channels=feature))

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        for idx in range(0, len(self.layers), 2):
            x = self.layers[idx](x)
            skip_con = skip_connections.pop()
            x = self.layers[idx + 1](torch.concat((skip_con, x), dim=1))

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding="same", bias=False
            ),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same", bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
