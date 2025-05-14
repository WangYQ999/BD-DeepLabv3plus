import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, num_patches=1024):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.num_patches = num_patches

        self.to_qkv = nn.Conv2d(
            dim, inner_dim * 3, kernel_size=1, padding=0, bias=False
        )

        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)
        self.sig = nn.Sigmoid()

        self.to_out = (
            nn.Sequential(
                nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(dim),  # inner_dim
                nn.ReLU(inplace=True),
            )
            if project_out
            else nn.Identity()
        )  # .to(device)

    def forward(self, x, mode="train", smooth=1e-4):
        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, "b (g d) h w -> b g (h w) d", g=self.heads), qkv
        )

        attn = torch.matmul(q, k.transpose(-1, -2))  # b g n n

        qk_norm = (
            torch.sqrt(torch.sum(q**2, dim=-1) + smooth)[:, :, :, None]
            * torch.sqrt(torch.sum(k**2, dim=-1) + smooth)[:, :, None, :]
            + smooth
        )
        attn = attn / qk_norm

        out = torch.matmul(attn, v)
        out = rearrange(out, "b g (h w) d -> b (g d) h w", h=x[0].shape[2])
        if mode == "train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn


# torch.nn.Conv2d(in_channels, out_channels,kernel_size,stride=1, padding=0, dilation=1, groups=1, bias=True)
class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# num_patches=h*w
class CNNTransformer_record(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim=256, dropout=0.0, num_patches=1024
    ):  # 4096
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CNNAttention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            num_patches=num_patches,
                        ),
                        CNNFeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)
        return x
