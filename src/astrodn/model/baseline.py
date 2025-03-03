"""Baseline denoiser model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding='valid'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class Up(nn.Module):
    """Upscaling using transposed convolution"""

    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x


class DownLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv2D(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = Up(in_ch)
        self.conv = DoubleConv2D(in_ch, out_ch)
    
    def forward(self, x1, x2):
        return self.conv(self.up(x1, x2))


class Baseline(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv1 = DoubleConv2D(in_ch, 64)
        self.down1 = DownLayer(64, 128)
        self.down2 = DownLayer(128, 256)
        self.down3 = DownLayer(256, 512) 

        self.up1 = UpLayer(512, 256)
        self.up2 = UpLayer(256, 128)
        self.up3 = UpLayer(128, 64)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)

        return self.outconv(x7)
