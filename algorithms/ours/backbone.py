import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class LightUNet(nn.Module):
    """Lightweight U-Net: 3-level encoder, channels [32, 64, 128].
    Input/output: (B, 1, H, W), H=W=64."""

    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels

        self.enc1 = ConvBnRelu(1,  c1)
        self.enc2 = ConvBnRelu(c1, c2)
        self.enc3 = ConvBnRelu(c2, c3)
        self.bot  = ConvBnRelu(c3, c3)

        self.dec3 = ConvBnRelu(c3 + c3, c2)
        self.dec2 = ConvBnRelu(c2 + c2, c1)
        self.dec1 = nn.Conv2d(c1 + c1, 1, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)                # (B, c1, 64, 64)
        e2 = self.enc2(self.pool(e1))    # (B, c2, 32, 32)
        e3 = self.enc3(self.pool(e2))    # (B, c3, 16, 16)
        b  = self.bot(self.pool(e3))     # (B, c3,  8,  8)

        d = F.interpolate(b,  scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec3(torch.cat([d, e3], dim=1))   # (B, c2, 16, 16)

        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([d, e2], dim=1))   # (B, c1, 32, 32)

        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec1(torch.cat([d, e1], dim=1))   # (B,  1, 64, 64)
        return d
