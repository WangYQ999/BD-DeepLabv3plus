import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        # 多尺度池化分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道注意力
        self.channel_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 多尺度池化
        avg_out = self.avg_pool(x).view(b, c)#[4,1280]
        max_out = self.max_pool(x).view(b, c)#[4,1280]

        # 通道注意力
        channel_weights = self.channel_fc(avg_out + max_out).view(b, c, 1, 1)
        out = x * channel_weights.expand_as(x)

        return out