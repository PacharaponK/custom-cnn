"""A small U-Net defined here and initialized from scratch."""
import torch
from torch import nn
from torch.nn import functional as F

def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )

class CompactUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = block(3, 8)
        self.enc2 = block(8, 16)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(16, 32)
        self.dec2 = block(48, 16)
        self.dec1 = block(24, 8)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([F.interpolate(b, size=e2.shape[-2:],
                         mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:],
                         mode="bilinear", align_corners=False), e1], dim=1))
        return self.head(d1)
