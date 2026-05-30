"""LoRA adapters for LightUNet, plus condition MLP for Stage 3.

Each LoRABank holds one pair (A, B) per Conv layer in LightUNet.
The effective weight at stage k is:  W_k = W_base + B_k @ A_k (reshaped).

A is zero-init, B is Kaiming-uniform → ΔW = 0 at the start of training.

ConditionMLP maps a scalar condition c^(k) to a denoiser scale s^(k) ∈ (0, 2).
Initialized to output 1.0 (identical to Stage 2 at the start).
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


class GateMLP(nn.Module):
    """Maps scalar residual energy e^(k) → gate g^(k) ∈ (0, 1).

    Initialized so g ≈ 1.0 at start (sigmoid(4) ≈ 0.982),
    meaning the model begins identical to LoRADUNDR and learns
    to selectively suppress LoRA where it is not needed.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)   # sigmoid(4) ≈ 0.982

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e : (B,) residual energy  →  gate : (B,) ∈ (0, 1)"""
        return torch.sigmoid(self.net(e.unsqueeze(-1))).squeeze(-1)


def unet_with_gated_lora(unet, lora: LoRABank,
                         x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """LightUNet forward with per-sample gated LoRA.

    gate : (B,) scalar ∈ (0,1) — applied to the LoRA feature-map delta
           so each sample can have a different LoRA contribution level.

    Separates base conv and LoRA delta in feature space (not weight space)
    because gate is per-sample and cannot be folded into a shared weight.
    """
    g = gate.view(-1, 1, 1, 1)   # (B, 1, 1, 1) for broadcasting

    def cbr(block, adapter, inp):
        conv, bn, relu = block[0], block[1], block[2]
        base  = F.conv2d(inp, conv.weight,          conv.bias, conv.stride, conv.padding)
        delta = F.conv2d(inp, adapter.delta_weight(), None,    conv.stride, conv.padding)
        return relu(bn(base + g * delta))

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
    inp_last = torch.cat([d, e1], dim=1)
    base  = F.conv2d(inp_last, unet.dec1.weight,           unet.dec1.bias,
                     unet.dec1.stride, unet.dec1.padding)
    delta = F.conv2d(inp_last, lora.dec1.delta_weight(), None,
                     unet.dec1.stride, unet.dec1.padding)
    return base + g * delta


class ConditionMLP(nn.Module):
    """Maps scalar condition c^(k) → denoiser scale s^(k) ∈ (0, 2).

    Initialized so that s ≡ 1.0 at the start (2·sigmoid(0) = 1),
    matching Stage 2 (LoRADUNFixed) behaviour exactly before training.
    The network then learns to attenuate or amplify denoising per stage
    based on the current reconstruction state.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)   # sigmoid(0) = 0.5 → 2×0.5 = 1.0

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c : (B,) scalar condition → (B,) scale in (0, 2)"""
        return 2.0 * torch.sigmoid(self.net(c.unsqueeze(-1))).squeeze(-1)
