import torch
from torch import nn, Tensor
import torch.nn.functional as F


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            (kernel_size, 1),
            (stride, 1),
            (padding, 0),
            bias=bias,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            (1, kernel_size),
            (1, stride),
            (0, padding),
            bias=bias,
        )

    def forward(self, x):

        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.branch1_bn = nn.BatchNorm2d(in_channels)
        self.branch1_relu = nn.ReLU()

        self.branch2a_conv = nn.Conv2d(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.branch2a_bn = nn.BatchNorm2d(mid_channels)
        self.branch2a_relu = nn.ReLU()

        self.branch2b_conv = nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False)

        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
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


class DWResBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()

        self.branch1_bn = nn.BatchNorm2d(in_channels)
        self.branch1_relu = nn.ReLU()

        self.branch2a_conv = DWConv(in_channels, mid_channels, 3, stride, 1, bias=False)
        self.branch2a_bn = nn.BatchNorm2d(mid_channels)
        self.branch2a_relu = nn.ReLU()

        self.branch2b_conv = DWConv(mid_channels, out_channels, 3, 1, 1, bias=False)

        if in_channels != out_channels:
            self.downsample = DWConv(in_channels, out_channels, 3, stride, 1, bias=False)
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


class ChannelAttention(nn.Module):
    def __init__(self, in_dim, embed_dim: int) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.embed_dim = embed_dim

        self.q_fc = nn.Linear(self.in_dim, self.embed_dim, bias=False)
        self.k_fc = nn.Linear(self.in_dim, self.embed_dim, bias=False)
        self.v_fc = nn.Linear(self.in_dim, self.embed_dim, bias=False)

        self.scale = self.embed_dim ** -0.5

        self.proj = nn.Linear(self.embed_dim, self.in_dim, bias=False)

    def forward(self, x):  # x: (B, C, H, W)

        B, C, H, W = x.shape

        z = x.mean(dim=(2, 3))  # (B, C)

        Q = self.q_fc(z).unsqueeze(-1)  # (B,1,D)
        K = self.k_fc(z).unsqueeze(-1)  # (B,1,D)
        V = self.v_fc(z).unsqueeze(-1)  # (B,1,D)
        # print("1", Q.shape, K.shape, V.shape)

        attn = Q @ K.transpose(-2, -1)
        attn = attn * self.scale
        attn = torch.softmax(attn, dim=-1)  # (B,1,1)
        # print("2", attn.shape)

        out = (attn @ V).squeeze(-1)
        out = self.proj(out)
        # print("3", out.shape)

        w = torch.sigmoid(out).view(B, C, 1, 1)
        # print("4", w.shape)

        return x * w

    # def forward(self, x: Tensor) -> Tensor:
    #     # x: B, C, H, W
    #
    #     x_avg = F.adaptive_avg_pool2d(x, (1, 1))
    #     x_max = F.adaptive_max_pool2d(x, (1, 1))
    #
    #     x_avg = self.mlp(x_avg)
    #     x_max = self.mlp(x_max)
    #
    #
    #
    #     B, C, H, W = x.shape
    #     N = H * W
    #
    #     q = self.query(x)  # (B, emb, H, W)
    #     k = self.key(x)  # (B, emb, H, W)
    #     v = self.value(x)  # (B, emb, H, W)
    #
    #     q = q.view(B, self.embed_dim, -1)
    #     k = k.view(B, self.embed_dim, -1)
    #     v = v.view(B, self.embed_dim, -1)
    #
    #     scores = q @ k.transpose(-2, -1) / (self.embed_dim ** 0.5)
    #     attn = self.softmax(scores)
    #
    #     out = attn @ v
    #
    #     out = out.reshape(B, self.embed_dim, H, W)
    #
    #     out = self.proj(out)
    #
    #     return out


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, sp_dim: int, embed_dim: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.query_norm_act = nn.Sequential(nn.BatchNorm1d(sp_dim), nn.ReLU())
        self.query_conv = nn.Linear(sp_dim, embed_dim, bias=False)

        self.key_norm_act = nn.Sequential(nn.BatchNorm1d(sp_dim), nn.ReLU())
        self.key_conv = nn.Linear(sp_dim, embed_dim, bias=False)

        self.value_norm_act = nn.Sequential(nn.BatchNorm1d(sp_dim), nn.ReLU())
        self.value_conv = nn.Linear(sp_dim, embed_dim, bias=False)

        self.proj_conv = nn.Linear(embed_dim, sp_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:

        # print(x.shape)
        B, C, H, W = x.shape
        N = H * W

        x = x.view(B, C, -1).transpose(1, 2)
        q = self.query_norm_act(x)
        k = self.key_norm_act(x)
        v = self.value_norm_act(x)
        # print(q.shape, k.shape, v.shape)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # print(q.shape, k.shape, v.shape)

        q = self.query_conv(q)
        k = self.key_conv(k)
        v = self.value_conv(v)

        # print(q.shape, k.shape, v.shape)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # print(q.shape, k.shape, v.shape)

        scores = q @ k.transpose(-2, -1) / (self.embed_dim ** 0.5)
        attn = self.softmax(scores)

        # print(attn.shape)

        out = attn @ v

        # print(out.shape)

        out = out.transpose(1, 2)
        # print(out.shape)

        out = self.proj_conv(out)
        # print(out.shape)

        out = out.reshape(B, C, H, W)

        return out

if __name__ == "__main__":
    Attn = ChannelAttention(32, 16)

    input = torch.randn(2, 32, 16, 16)

    out = Attn(input)
    print(out.shape)