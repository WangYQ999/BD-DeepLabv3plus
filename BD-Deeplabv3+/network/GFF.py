import torch
import torch.nn as nn


class GatedFeatureFusion(nn.Module):
    def __init__(self, in_channels_A, in_channels_B, out_channels):
        super(GatedFeatureFusion, self).__init__()
        # 1×1 卷积进行通道对齐
        self.conv_A = nn.Conv2d(in_channels_A, out_channels, kernel_size=1, bias=False)
        self.conv_B = nn.Conv2d(in_channels_B, out_channels, kernel_size=1, bias=False)

        # 计算门控权重
        self.gate = nn.Conv2d(out_channels * 2, 2, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_A, feat_B):
        feat_A = self.conv_A(feat_A)  # (B, out_channels, H, W)
        feat_B = self.conv_B(feat_B)  # (B, out_channels, H, W)

        # 计算门控权重
        fusion_weight = self.sigmoid(
            self.gate(torch.cat([feat_A, feat_B], dim=1))
        )  # (B, 2, H, W)
        gate_A, gate_B = torch.split(fusion_weight, 1, dim=1)  # (B, 1, H, W)

        # 进行门控融合
        fused_feature = gate_A * feat_A + gate_B * feat_B
        return fused_feature
