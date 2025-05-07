import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ._blocks import ResidualBlock, ChannelAttention, SpatialAttention


class Try4Model(nn.Module):
    def __init__(self):
        super(Try4Model, self).__init__()

        self.block1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.block1_ch_attn = ChannelAttention(8, 8)
        # self.block1_sp_attn = SpatialAttention(8, 32 * 32, 1024)

        self.block2 = ResidualBlock(8, 16, 8, 1)
        self.block2_ch_attn = ChannelAttention(8, 8)
        # self.block2_sp_attn = SpatialAttention(8, 32 * 32, 1024)

        self.block3 = ResidualBlock(8, 16, 16, 2)
        self.block3_ch_attn = ChannelAttention(16, 16)
        # self.block3_sp_attn = SpatialAttention(16, 16 * 16, 256)

        self.block4 = ResidualBlock(16, 32, 16, 1)
        self.block4_ch_attn = ChannelAttention(16, 16)
        # self.block4_sp_attn = SpatialAttention(16, 16 * 16, 256)

        self.block5 = ResidualBlock(16, 32, 32, 2)
        self.block5_ch_attn = ChannelAttention(32, 32)
        # self.block5_sp_attn = SpatialAttention(32, 8 * 8, 16)

        self.block6 = ResidualBlock(32, 64, 32, 1)
        self.block6_ch_attn = ChannelAttention(32, 32)
        # self.block6_sp_attn = SpatialAttention(32, 8 * 8, 16)

        self.block7 = ResidualBlock(32, 64, 64, 2)
        self.block7_ch_attn = ChannelAttention(64, 64)
        # self.block7_sp_attn = SpatialAttention(64, 4 * 4, 8)

        self.block8 = ResidualBlock(64, 128, 64, 1)
        self.block8_ch_attn = ChannelAttention(64, 64)
        # self.block8_sp_attn = SpatialAttention(64, 4 * 4, 8)

        self.block9_bn = nn.BatchNorm2d(64)
        self.block9_relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

        self.num_params = sum(p.numel() for p in list(self.parameters()))
        print(f"# params: {self.num_params}")

    def forward(self, x):

            x = self.block1(x)
            x = self.block1_ch_attn(x)
            # x = self.block1_sp_attn(x)

            x = self.block2(x)
            x = self.block2_ch_attn(x)
            # x = self.block2_sp_attn(x)

            x = self.block3(x)
            x = self.block3_ch_attn(x)
            # x = self.block3_sp_attn(x)

            x = self.block4(x)
            x = self.block4_ch_attn(x)
            # x = self.block4_sp_attn(x)

            x = self.block5(x)
            x = self.block5_ch_attn(x)
            # x = self.block5_sp_attn(x)

            x = self.block6(x)
            x = self.block6_ch_attn(x)
            # x = self.block6_sp_attn(x)

            x = self.block7(x)
            x = self.block7_ch_attn(x)
            # x = self.block7_sp_attn(x)

            x = self.block8(x)
            x = self.block8_ch_attn(x)
            # x = self.block8_sp_attn(x)

            x = self.block9_bn(x)
            x = self.block9_relu(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x
