import torch
from torch import nn, Tensor


class ConvNormActDrop(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        drop_rate=0.0,

    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=drop_rate)

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x

