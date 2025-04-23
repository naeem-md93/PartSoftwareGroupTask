import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .blocks import ConvNormActDrop


class Try2Model(nn.Module):
    def __init__(self):
        super(Try2Model, self).__init__()

        self.block1 = ConvNormActDrop(3, 8, 3)
        self.block4 = ConvNormActDrop(8, 16, 3, 2, 1)
        self.block7 = ConvNormActDrop(16, 32, 3, 2, 1)
        self.block10 = ConvNormActDrop(32, 64, 3, 2, 1)
        self.block13 = ConvNormActDrop(64, 128, 3, 2, 1)
        self.block14 = ConvNormActDrop(128, 256, 3, 2, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

        self.num_params = sum(p.numel() for p in list(self.parameters()))
        print(f"# params: {self.num_params}")

    def forward(self, x):

        x = self.block1(x)
        x = self.block4(x)
        x = self.block7(x)
        x = self.block10(x)
        x = self.block13(x)
        x = self.block14(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
