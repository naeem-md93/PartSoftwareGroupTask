import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .blocks import ResidualBlock, DWResidualBlock


class Try3Model(nn.Module):
    def __init__(self):
        super(Try3Model, self).__init__()

        self.block1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.block2 = ResidualBlock(8, 16, 8)

        self.block3 = DWResidualBlock(8, 16, 16)
        self.block4 = ResidualBlock(16, 32, 16)

        self.block5 = DWResidualBlock(16, 32, 32)
        self.block6 = ResidualBlock(32, 64, 32)

        self.block7 = DWResidualBlock(32, 64, 64)
        self.block8 = ResidualBlock(64, 128, 64)

        self.block9_bn = nn.BatchNorm2d(64)
        self.block9_relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

        self.num_params = sum(p.numel() for p in list(self.parameters()))
        print(f"# params: {self.num_params}")

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)

        x = self.block5(x)
        x = self.block6(x)

        x = self.block7(x)
        x = self.block8(x)

        x = self.block9_bn(x)
        x = self.block9_relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
