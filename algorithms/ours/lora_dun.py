import torch
import torch.nn as nn

from backbone import LightUNet
from unfolding import ISTAStep
from lora import LoRABank, unet_with_lora


class BaseDUN(nn.Module):
    """Stage 0: each unrolled stage has its own independent U-Net.
    No parameter sharing, no LoRA.  ~4.3 M params at 10 stages."""

    def __init__(self, num_stages: int = 10, channels=(32, 64, 128),
                 patch_size: int = 64):
        super().__init__()
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.stages = nn.ModuleList(
            [ISTAStep(LightUNet(channels)) for _ in range(num_stages)]
        )

    def forward(self, y: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """
        y   : (B, M)  measurements
        Phi : (M, N)  measurement matrix, N = patch_size^2
        Returns (B, 1, H, W) reconstructed patch.
        """
        H = W = self.patch_size
        x = y @ Phi     # back-projection init: x^0 = Phi^T y  (B, N)
        for stage in self.stages:
            x = stage(x, y, Phi, H, W)
        return x.view(y.shape[0], 1, H, W)


class SharedDUN(nn.Module):
    """Stage 1: all unrolled stages share ONE U-Net (full weight sharing).
    Each stage keeps its own learnable step-size alpha.
    ~0.43 M params total (10x reduction vs Base-DUN)."""

    def __init__(self, num_stages: int = 10, channels=(32, 64, 128),
                 patch_size: int = 64):
        super().__init__()
        self.num_stages = num_stages
        self.patch_size = patch_size
        shared_net = LightUNet(channels)
        self.stages = nn.ModuleList(
            [ISTAStep(shared_net) for _ in range(num_stages)]
        )

    def forward(self, y: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        H = W = self.patch_size
        x = y @ Phi
        for stage in self.stages:
            x = stage(x, y, Phi, H, W)
        return x.view(y.shape[0], 1, H, W)


class LoRADUNFixed(nn.Module):
    """Stage 2: shared backbone + stage-specific LoRA adapters (fixed rank).

    All 10 stages share ONE LightUNet backbone. Each stage additionally has
    its own LoRABank that adds ΔW_k = B_k @ A_k to every Conv layer.
    The effective denoiser at stage k is: W_base + ΔW_k.

    Parameter budget (rank=4, channels=[32,64,128]):
        backbone  : ~0.43 M  (shared, counted once)
        LoRA banks: 10 × ~26 K ≈ 0.26 M
        alphas    : 10 scalars
        total     : ~0.69 M  (vs 4.3 M for Base-DUN)
    """

    def __init__(self, num_stages: int = 10, channels=(32, 64, 128),
                 patch_size: int = 64, rank: int = 4):
        super().__init__()
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.backbone   = LightUNet(channels)
        self.lora_banks = nn.ModuleList(
            [LoRABank(channels, rank) for _ in range(num_stages)]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_stages)]
        )

    def forward(self, y: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        H = W = self.patch_size
        B = y.shape[0]
        x = y @ Phi  # back-projection init

        for k in range(self.num_stages):
            Phix = x @ Phi.T
            grad = (Phix - y) @ Phi
            r    = x - self.alphas[k] * grad
            x    = unet_with_lora(self.backbone, self.lora_banks[k],
                                  r.view(B, 1, H, W)).view(B, H * W).clamp(0., 1.)

        return x.view(B, 1, H, W)


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get('model_name', 'BaseDUN')
    kwargs = dict(
        num_stages=cfg.get('num_stages', 10),
        channels=tuple(cfg.get('channels', [32, 64, 128])),
        patch_size=cfg.get('patch_size', 64),
    )
    if name == 'BaseDUN':
        return BaseDUN(**kwargs)
    if name == 'SharedDUN':
        return SharedDUN(**kwargs)
    if name == 'LoRADUNFixed':
        return LoRADUNFixed(**kwargs, rank=cfg.get('lora_rank', 4))
    raise ValueError(f'Unknown model: {name}')
