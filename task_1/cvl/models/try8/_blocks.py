import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.branch1_bn = nn.BatchNorm2d(in_channels)
        self.branch1_relu = nn.ReLU()

        self.branch2_1x1 = nn.Conv2d(in_channels, mid_channels, 1, stride, bias=False)
        self.branch2_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, padding=1, bias=False)
        )
        self.branch2_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        )

        self.branch2_7x7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        )

        if stride == 1:
            self.branch2_pool = nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
                nn.Conv2d(in_channels, mid_channels, 1, 1, bias=False)
            )
        else:
            self.branch2_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(in_channels, mid_channels, 1, 1, bias=False)
            )

        self.branch2_bn = nn.BatchNorm2d(mid_channels * 5)
        self.branch2_relu = nn.ReLU()

        self.branch2_proj = nn.Conv2d(mid_channels * 5, out_channels, 1, bias=False)

        if (in_channels != out_channels) or (stride == 2):
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, branch1: Tensor) -> Tensor:

        branch1 = self.branch1_bn(branch1)
        branch1 = self.branch1_relu(branch1)

        b2_1x1 = self.branch2_1x1(branch1)
        b2_3x3 = self.branch2_3x3(branch1)
        b2_5x5 = self.branch2_5x5(branch1)
        b2_7x7 = self.branch2_7x7(branch1)
        b2_pool = self.branch2_pool(branch1)

        branch2 = torch.cat([b2_1x1, b2_3x3, b2_5x5, b2_7x7, b2_pool], dim=1)

        branch2 = self.branch2_bn(branch2)
        branch2 = self.branch2_relu(branch2)

        branch2 = self.branch2_proj(branch2)

        branch2 = branch2 + self.downsample(branch1)

        return branch2


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int, dropout_rate: float = 0.0) -> None:
        super().__init__()

        self.branch1_bn = nn.BatchNorm2d(in_channels)
        self.branch1_relu = nn.ReLU()

        self.branch2a_conv = nn.Conv2d(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.branch2a_bn = nn.BatchNorm2d(mid_channels)
        self.branch2a_relu = nn.ReLU()
        self.branch2a_dropout = nn.Dropout2d(dropout_rate)

        self.branch2b_conv = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)

        if (in_channels != out_channels) or (stride == 2):
            self.downsample = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.downsample = nn.Identity()

    def forward(self, branch1: Tensor) -> Tensor:

        branch1 = self.branch1_bn(branch1)
        branch1 = self.branch1_relu(branch1)

        branch2 = self.branch2a_conv(branch1)
        branch2 = self.branch2a_bn(branch2)
        branch2 = self.branch2a_relu(branch2)
        branch2 = self.branch2a_dropout(branch2)

        branch2 = self.branch2b_conv(branch2)

        branch2 = branch2 + self.downsample(branch1)

        return branch2


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, num_heads: int = 4, dropout_rate: float = 0.0) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_fc = nn.Linear(in_channels, embed_dim, bias=False)
        self.k_fc = nn.Linear(in_channels, embed_dim, bias=False)
        self.v_fc = nn.Linear(in_channels, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, in_channels, bias=False)

    def forward(self, x):  # x: (B, C, H, W)

        B, C, H, W = x.shape
        N = H * W
        # assert N % self.num_heads == 0, f"Number of heads {self.num_heads} is not divisible by embedding {N}"
        head_dim = N // self.num_heads
        scale = head_dim ** -0.5

        desc = x.view(B, C, -1).transpose(-2, -1)  # (B, N, C)

        Q = self.q_fc(desc).transpose(-2, -1)  # (B, C, E)
        K = self.k_fc(desc).transpose(-2, -1)  # (B, C, E)
        V = self.v_fc(desc).transpose(-2, -1)  # (B, C, E)

        Q = Q.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, head_dim).transpose(1, 2)

        attn_scores = Q @ K.transpose(-1, -2) * scale
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # print(attn_probs.shape)
        attn_probs = self.attn_dropout(attn_probs)

        attn_out = attn_probs @ V
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, C, -1).mean(dim=-1)

        out = self.proj(attn_out)  # (B, C)

        w = torch.sigmoid(out).view(B, C, 1, 1)

        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, window_size: int, embed_dim: int, patch_size: int, num_heads: int = 4, dropout_rate: float = 0.0) -> None:
        super().__init__()

        assert window_size % patch_size == 0, "window_size must be divisible by patch_size"
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.in_channels = in_channels
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (window_size // patch_size) ** 2
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Linear projections for Q, K, V
        self.q_fc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_fc = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_fc = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:

        # Patch embedding: (B, embed_dim, P, P)
        patches = self.patch_embed(x)

        # Flatten patches: (B, N, embed_dim)
        patches = patches.flatten(2).transpose(1, 2)  # N = num_patches

        # Add positional embedding
        patches = patches + self.pos_embed

        # QKV projections
        Q = self.q_fc(patches)  # (B, N, E)
        K = self.k_fc(patches)
        V = self.v_fc(patches)

        Q = Q.view(x.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(x.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(x.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_out = attn_probs @ V

        attn_out = attn_out.transpose(1, 2).reshape(x.shape[0], self.num_patches, self.embed_dim)

        attn_out = self.proj(attn_out)  # (B, N, E)

        attn_out = attn_out.mean(dim=-1, keepdim=True)

        out = attn_out.reshape(x.shape[0], 1, self.window_size // self.patch_size, self.window_size // self.patch_size)

        attn_map = torch.sigmoid(out)

        attn_map = F.interpolate(attn_map, scale_factor=self.patch_size, mode='nearest')

        return x * attn_map

if __name__ == "__main__":
    Attn = SpatialAttention(32, 16, 32, 4, 4, 0.1)

    input = torch.randn(2, 32, 16, 16)

    out = Attn(input)
    print(out.shape)