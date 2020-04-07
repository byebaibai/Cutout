import torch
import torch.nn as nn
import torch.nn.functional as F


def ConvBnReLU6(in_channels, out_channels, stride, kernel_size=3, padding=1, groups=1, relu6=True):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels))
    if relu6:
        block.add_module('relu6', nn.ReLU6(inplace=True))
    return block


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.useShortcut = stride == 1 and in_channels == out_channels
        self.convBlock = nn.Sequential(
            # pw
            ConvBnReLU6(in_channels, in_channels * expand_ratio, kernel_size=1, stride=1, padding=0),
            # dw
            ConvBnReLU6(in_channels * expand_ratio, in_channels * expand_ratio, kernel_size=3, stride=stride,
                        padding=1, groups=in_channels * expand_ratio),
            # pw-linear
            ConvBnReLU6(in_channels * expand_ratio, out_channels, kernel_size=1, stride=1, padding=0, relu6=False),
        )

    def forward(self, x):
        if self.useShortcut:
            return x + self.convBlock(x)
        else:
            return self.convBlock(x)


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(MobileNetBlock, self).__init__()
        self.conv_head = ConvBnReLU6(in_channels, 32, stride=2)

        self.block_1 = InvertedResidual(32, 16, 1, 1)
        # 1/4 128 x 128
        self.block_2 = nn.Sequential(
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6)
        )
        # 1/8 64 x 64
        self.block_3 = nn.Sequential(
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6)
        )
        # 1/16 32 x 32
        self.block_4 = nn.Sequential(
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
        )
        # 1/16 32 x 32
        self.block_5 = nn.Sequential(
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
        )
        # 1/32 16 x 16
        self.block_6 = nn.Sequential(
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6)
        )
        # 1/32 16 x 16
        self.block_7 = InvertedResidual(160, 320, 1, 6)

    def forward(self, input):
        x = self.conv_head(input)

        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)
        x4 = self.block_5(x4)
        x5 = self.block_6(x4)
        x5 = self.block_7(x5)

        return x1, x2, x3, x4, x5


class UetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(UetBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.aspp = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels * 2)
        )

    def forward(self, e, x):
        x = self.up(x)
        x = torch.cat((x, e), dim=1)
        x = self.aspp(x)

        return x

