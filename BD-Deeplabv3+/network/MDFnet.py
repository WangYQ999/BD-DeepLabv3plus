import torch
import torch.nn as nn
import torch.nn.functional as F


class MDFNet(nn.Module):
    def __init__(self, in_channels=[24, 64, 160], out_channels=256):
        super(MDFNet, self).__init__()

        # 1. 高层和中层特征加入空洞卷积增强
        self.high_dilated = nn.Conv2d(
            in_channels[2],
            in_channels[2],
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=in_channels[2],
        )
        self.middle_dilated = nn.Conv2d(
            in_channels[1],
            in_channels[1],
            kernel_size=3,
            padding=1,
            dilation=1,
            groups=in_channels[1],
        )

        # 2. 通道注意力模块
        self.channel_attention_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[2], in_channels[2] // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels[2] // 8, in_channels[2], kernel_size=1),
            nn.Sigmoid(),
        )

        self.channel_attention_middle = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels[1], in_channels[1] // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels[1] // 8, in_channels[1], kernel_size=1),
            nn.Sigmoid(),
        )

        # 3. 上采样模块（带有跨尺度引导）
        self.upsample_high = nn.ConvTranspose2d(
            in_channels[2], in_channels[2], kernel_size=4, stride=4, padding=0
        )
        self.upsample_middle = nn.ConvTranspose2d(
            in_channels[1], in_channels[1], kernel_size=4, stride=4, padding=0
        )

        # 4. 最终融合
        self.final_conv = nn.Conv2d(
            in_channels[0] + in_channels[1] + in_channels[2],
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x_low, x_middle, x_high):
        # 高级特征加入空洞卷积
        x_high = self.high_dilated(x_high) * self.channel_attention_high(x_high)

        # 中级特征加入空洞卷积
        x_middle = self.middle_dilated(x_middle) * self.channel_attention_middle(
            x_middle
        )

        # 高层特征上采样
        x_high_up = self.upsample_high(x_high)

        x_middle_up = self.upsample_middle(x_middle)

        # 拼接
        fused = torch.cat([x_low, x_middle_up, x_high_up], dim=1)

        # 最终融合
        output = self.final_conv(fused)

        return output
