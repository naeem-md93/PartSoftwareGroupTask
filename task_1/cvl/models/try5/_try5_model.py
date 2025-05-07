import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ._blocks import ResidualBlock, ChannelAttention, SpatialAttention, DWResBlock


class Try5Model(nn.Module):
    def __init__(self):
        super(Try5Model, self).__init__()

        # ===========================
        self.block1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)  # 32 x 32
        self.block1_ch_attn = ChannelAttention(8, 8)

        self.block2 = ResidualBlock(8, 16, 8)                  # 32 x 32
        self.block2_ch_attn = ChannelAttention(8, 8)
        # ===========================

        # ===========================
        self.block3 = ResidualBlock(8, 16, 16, 2)        # 16 x 16
        self.block3_ch_attn = ChannelAttention(16, 16)

        self.block4 = ResidualBlock(16, 32, 16)                # 16 x 16
        self.block4_ch_attn = ChannelAttention(16, 16)
        # ===========================

        # ===========================
        self.block5 = ResidualBlock(16, 32, 32, 2)       # 8 x 8
        self.block5_ch_attn = ChannelAttention(32, 32)

        self.block6 = ResidualBlock(32, 64, 32)                # 8 x 8
        self.block6_ch_attn = ChannelAttention(32, 32)
        # ===========================

        # ===========================
        self.block7 = DWResBlock(32, 64, 64, 2)          # 4 x 4
        self.block7_ch_attn = ChannelAttention(64, 64)

        self.block8 = DWResBlock(64, 128, 128, 2)        # 2 x 2
        # ===========================

        self.block9_bn = nn.BatchNorm2d(128)
        self.block9_relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)

        self.num_params = sum(p.numel() for p in list(self.parameters()))
        print(f"# params: {self.num_params}")

    def forward(self, x):

        x = self.block1(x)
        x = self.block1_ch_attn(x)

        x = self.block2(x)
        x = self.block2_ch_attn(x)

        x = self.block3(x)
        x = self.block3_ch_attn(x)

        x = self.block4(x)
        x = self.block4_ch_attn(x)

        x = self.block5(x)
        x = self.block5_ch_attn(x)

        x = self.block6(x)
        x = self.block6_ch_attn(x)

        x = self.block7(x)
        x = self.block7_ch_attn(x)

        x = self.block8(x)

        x = self.block9_bn(x)
        x = self.block9_relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
