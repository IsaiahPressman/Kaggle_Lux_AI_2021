from typing import Tuple
import torch
from torch import nn

from .conv_blocks import ResidualBlock


class UNET(nn.Module):
    def __init__(
            self,
            n_blocks_per_reduction: int,
            in_out_channels: int,
            height: int,
            width: int,
            **residual_block_kwargs
    ):
        super(UNET, self).__init__()

        block1_channels = in_out_channels
        block1_h = height
        block1_w = width

        block2_channels = 2 * in_out_channels
        block2_h = height // 2
        block2_w = width // 2

        block3_channels = 4 * in_out_channels
        block3_h = height // 4
        block3_w = width // 4
        
        self.block1_down = nn.Sequential(
            *[ResidualBlock(
                in_channels=block1_channels,
                out_channels=block1_channels,
                height=block1_h,
                width=block1_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction)]
        )
        self.reduction1 = nn.AvgPool2d(kernel_size=2)
        self.expansion1 = nn.ConvTranspose2d(block2_channels, block1_channels, kernel_size=(2, 2), stride=(2, 2))
        self.block1_up = nn.Sequential(
            ResidualBlock(
                in_channels=block2_channels,
                out_channels=block1_channels,
                height=block1_h,
                width=block1_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block1_channels,
                out_channels=block1_channels,
                height=block1_h,
                width=block1_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )

        self.block2_down = nn.Sequential(
            ResidualBlock(
                in_channels=block1_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block2_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )
        self.reduction2 = nn.AvgPool2d(kernel_size=2)
        self.expansion2 = nn.ConvTranspose2d(block3_channels, block2_channels, kernel_size=(2, 2), stride=(2, 2))
        self.block2_up = nn.Sequential(
            ResidualBlock(
                in_channels=block3_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block2_channels,
                out_channels=block2_channels,
                height=block2_h,
                width=block2_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )

        self.block3 = nn.Sequential(
            ResidualBlock(
                in_channels=block2_channels,
                out_channels=block3_channels,
                height=block3_h,
                width=block3_w,
                **residual_block_kwargs
            ),
            *[ResidualBlock(
                in_channels=block3_channels,
                out_channels=block3_channels,
                height=block3_h,
                width=block3_w,
                **residual_block_kwargs
            ) for _ in range(n_blocks_per_reduction - 1)]
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x_orig, input_mask_orig = x

        x1, input_mask1 = self.block1_down((x_orig, input_mask_orig.float()))
        x2, input_mask2 = self.block2_down((self.reduction1(x1), self.reduction1(input_mask1)))
        x3, _ = self.block3((self.reduction2(x2), self.reduction2(input_mask2)))

        x_out = torch.cat([x2, self.expansion2(x3)], dim=-3) * input_mask2
        x_out, _ = self.block2_up((x_out, input_mask2))
        x_out = torch.cat([x1, self.expansion1(x_out)], dim=-3) * input_mask1
        x_out, _ = self.block1_up((x_out, input_mask_orig))

        return x_out, input_mask_orig
