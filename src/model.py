import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(5, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
        )

        self.block1 = ResidualBlock(
            512,
            256,
            256,
        )
        self.block2 = ResidualBlock(
            256,
            128,
            128,
        )
        self.block3 = ResidualBlock(
            128,
            64,
            64,
        )
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.ds = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor):
        res = self.ds(x.clone())
        x = self.layers(x)
        x += res
        return x
