"""LoRA adapters for LightUNet.

Each LoRABank holds one pair (A, B) per Conv layer in LightUNet.
The effective weight at stage k is:  W_k = W_base + B_k @ A_k (reshaped).

A is zero-init, B is Kaiming-uniform → ΔW = 0 at the start of training.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv(nn.Module):
    """LoRA adapter for one Conv2d(in_ch, out_ch, k×k).
    delta_weight() returns a (out_ch, in_ch, k, k) tensor."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, rank: int = 4):
        super().__init__()
        fan_in = in_ch * kernel_size * kernel_size
        self.A = nn.Parameter(torch.zeros(rank, fan_in))
        self.B = nn.Parameter(torch.empty(out_ch, rank))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        self._shape = (out_ch, in_ch, kernel_size, kernel_size)

    def delta_weight(self) -> torch.Tensor:
        return (self.B @ self.A).view(self._shape)


class LoRABank(nn.Module):
    """One set of LoRA adapters for all 7 Conv layers in LightUNet."""

    def __init__(self, channels=(32, 64, 128), rank: int = 4):
        super().__init__()
        c1, c2, c3 = channels
        self.enc1 = LoRAConv(1,        c1,      3, rank)
        self.enc2 = LoRAConv(c1,       c2,      3, rank)
        self.enc3 = LoRAConv(c2,       c3,      3, rank)
        self.bot  = LoRAConv(c3,       c3,      3, rank)
        self.dec3 = LoRAConv(c3 + c3,  c2,      3, rank)
        self.dec2 = LoRAConv(c2 + c2,  c1,      3, rank)
        self.dec1 = LoRAConv(c1 + c1,  1,       3, rank)


def unet_with_lora(unet, lora: LoRABank, x: torch.Tensor) -> torch.Tensor:
    """LightUNet forward with LoRA delta weights added to each Conv layer.

    unet  : a LightUNet instance (shared backbone)
    lora  : LoRABank for this stage
    x     : (B, 1, H, W)
    """

    def cbr(block, adapter, inp):
        conv, bn, relu = block[0], block[1], block[2]
        w = conv.weight + adapter.delta_weight()
        return relu(bn(F.conv2d(inp, w, conv.bias, conv.stride, conv.padding)))

    # Encoder
    e1 = cbr(unet.enc1, lora.enc1, x)
    e2 = cbr(unet.enc2, lora.enc2, unet.pool(e1))
    e3 = cbr(unet.enc3, lora.enc3, unet.pool(e2))
    b  = cbr(unet.bot,  lora.bot,  unet.pool(e3))

    # Decoder
    d = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=False)
    d = cbr(unet.dec3, lora.dec3, torch.cat([d, e3], dim=1))

    d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
    d = cbr(unet.dec2, lora.dec2, torch.cat([d, e2], dim=1))

    d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
    w = unet.dec1.weight + lora.dec1.delta_weight()
    d = F.conv2d(torch.cat([d, e1], dim=1), w, unet.dec1.bias,
                 unet.dec1.stride, unet.dec1.padding)
    return d
