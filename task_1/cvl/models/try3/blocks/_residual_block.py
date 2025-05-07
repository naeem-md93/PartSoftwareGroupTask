import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ._conv_norm_act_drop import ConvNormActDrop


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()

        self.branch1_bn = nn.BatchNorm2d(in_channels)
        self.branch1_relu = nn.ReLU(inplace=True)

        self.branch2a_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.branch2a_bn = nn.BatchNorm2d(mid_channels)
        self.branch2a_relu = nn.ReLU(inplace=True)

        self.branch2b_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, branch1: Tensor) -> Tensor:

        branch1 = self.branch1_bn(branch1)
        branch1 = self.branch1_relu(branch1)

        branch2 = self.branch2a_conv(branch1)
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_relu(branch2)
        branch2 = self.branch2b_conv(branch2)

        branch2 = branch2 + self.downsample(branch1)

        return branch2


class DWResidualBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()

        self.branch1_bn = nn.BatchNorm2d(in_channels)
        self.branch1_relu = nn.ReLU(inplace=True)

        self.branch2a_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.branch2a_bn = nn.BatchNorm2d(mid_channels)
        self.branch2a_relu = nn.ReLU(inplace=True)

        self.branch2b_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, branch1: Tensor) -> Tensor:

        branch1 = self.branch1_bn(branch1)
        branch1 = self.branch1_relu(branch1)

        branch2 = self.branch2a_conv(branch1)
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_relu(branch2)
        branch2 = self.branch2b_conv(branch2)

        branch2 = branch2 + self.downsample(branch1)

        return branch2