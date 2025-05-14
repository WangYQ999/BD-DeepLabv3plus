import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedSEBlock(nn.Module):
    def __init__(self, channel, reduction=16, use_spatial_attention=True):
        super(ImprovedSEBlock, self).__init__()
        self.use_spatial_attention = use_spatial_attention

        # 多尺度池化分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道注意力
        self.channel_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

        # 空间注意力
        if self.use_spatial_attention:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), nn.Sigmoid()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        # 多尺度池化
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        # 通道注意力
        channel_weights = self.channel_fc(avg_out + max_out).view(b, c, 1, 1)
        out = x * channel_weights.expand_as(x)

        # 空间注意力
        if self.use_spatial_attention:
            avg_out_spatial = torch.mean(out, dim=1, keepdim=True)
            max_out_spatial, _ = torch.max(out, dim=1, keepdim=True)
            spatial_weights = self.spatial_conv(
                torch.cat([avg_out_spatial, max_out_spatial], dim=1)
            )
            out = out * spatial_weights

        return out
