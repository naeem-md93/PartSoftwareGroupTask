import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ._blocks import InceptionResidualBlock, ChannelAttention, SpatialAttention, DWResBlock, ResBlock


class Try7Model(nn.Module):
    def __init__(self):
        super(Try7Model, self).__init__()

        # 32 x 32 ===========================
        self.block1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.block2 = InceptionResidualBlock(8, 16, 8, 1)
        # ===================================

        # 16 x 16 ===========================
        self.block3 = InceptionResidualBlock(8, 16, 16, 2)

        self.block4 = nn.Sequential(
            InceptionResidualBlock(16, 32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.block4_ch_attn = ChannelAttention(16, 16, 4, 0.05)
        self.block4_sp_attn = SpatialAttention(16, 16, 16, 4, 4, 0.05)
        # ===================================

        # 8 x 8 =============================
        self.block5 = nn.Sequential(
            ResBlock(16, 32, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )                                                                                                       # 8 x 8
        self.block5_ch_attn = ChannelAttention(32, 32, 4, 0.1)
        self.block5_sp_attn = SpatialAttention(32, 8, 32, 2, 4, 0.1)

        # -----------------------------------

        self.block6 = nn.Sequential(
            ResBlock(32, 64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.block6_ch_attn = ChannelAttention(32, 32, 4, 0.1)
        self.block6_sp_attn = SpatialAttention(32, 8, 32, 2, 4, 0.1)
        self.block6_aux = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(32, 10)
        )
        self.block6_aux_coeff = nn.Parameter(torch.ones(1))
        # ===========================

        # 4 x 4 ===========================
        self.block7 = nn.Sequential(
            ResBlock(32, 64, 64, 2, 0.2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block7_ch_attn = ChannelAttention(64, 64, 4, 0.2)
        self.block7_sp_attn = SpatialAttention(64, 4, 64, 1, 4, 0.2)
        self.block7_aux = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(64, 10)
        )
        self.block7_aux_coeff = nn.Parameter(torch.ones(1))
        # 2 x 2 ===========================
        self.block8 = ResBlock(64, 64, 128, 2, 0.3)        # 2 x 2
        self.block8_aux = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 10)
        )
        self.block8_aux_coeff = nn.Parameter(torch.ones(1))


        self.num_params = sum(p.numel() for p in list(self.parameters()))
        print(f"# params: {self.num_params}")

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)

        x = self.block3(x)
        x = self.block4(x)
        x = x + self.block4_ch_attn(x)
        x = x + self.block4_sp_attn(x)

        x = self.block5(x)
        x = x + self.block5_ch_attn(x)
        x = x + self.block5_sp_attn(x)

        x = self.block6(x)
        x = x + self.block6_ch_attn(x)
        x = x + self.block6_sp_attn(x)
        out6 = self.block6_aux(x)

        x = self.block7(x)
        x = x + self.block7_ch_attn(x)
        x = x + self.block7_sp_attn(x)
        out7 = self.block7_aux(x)

        x = self.block8(x)
        out8 = self.block8_aux(x)

        return (out6 * self.block6_aux_coeff) + (out7 * self.block7_aux_coeff) + (out8 * self.block8_aux_coeff)
