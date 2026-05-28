import torch
import torch.nn as nn

from backbone import LightUNet
from unfolding import ISTAStep


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

        # Back-projection initialisation: x^0 = Phi^T y
        x = y @ Phi     # (B, N)

        for stage in self.stages:
            x = stage(x, y, Phi, H, W)

        return x.view(y.shape[0], 1, H, W)


def build_model(cfg: dict) -> nn.Module:
    name = cfg.get('model_name', 'BaseDUN')
    if name == 'BaseDUN':
        return BaseDUN(
            num_stages=cfg.get('num_stages', 10),
            channels=tuple(cfg.get('channels', [32, 64, 128])),
            patch_size=cfg.get('patch_size', 64),
        )
    raise ValueError(f'Unknown model: {name}')
