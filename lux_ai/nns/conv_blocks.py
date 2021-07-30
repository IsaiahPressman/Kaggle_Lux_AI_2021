from typing import *
import torch
from torch import nn


class SELayer(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FullConvResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            height: int,
            width: int,
            normalize: bool = False,
            activation: Callable = nn.ReLU,
            padding: Union[str, int, tuple[int, int]] = 'same',
            squeeze_excitation: bool = True,
            **conv2d_kwargs
    ):
        super(FullConvResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            **conv2d_kwargs
        )
        # We use LayerNorm here since the size of the input "images" may vary based on the board size
        self.norm1 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.act1 = activation()

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=padding,
            **conv2d_kwargs
        )
        self.norm2 = nn.LayerNorm([in_channels, height, width]) if normalize else nn.Identity()
        self.final_act = activation()

        if in_channels != out_channels:
            self.change_n_channels = nn.Conv2d(in_channels, out_channels, (1, 1))
        else:
            self.change_n_channels = nn.Identity()

        if squeeze_excitation:
            self.squeeze_excitation = SELayer(out_channels)
        else:
            self.squeeze_excitation = nn.Identity()

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        identity = x
        x = self.conv1(x) * input_mask
        x = self.act1(self.norm1(x))
        x = self.conv2(x) * input_mask
        x = self.squeeze_excitation(self.norm2(x))
        x = x + self.change_n_channels(identity)
        return self.final_activation(x) * input_mask, input_mask
