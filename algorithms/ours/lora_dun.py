import torch
import torch.nn as nn

from backbone import LightUNet
from unfolding import ISTAStep
from lora import LoRABank, ConditionMLP, GateMLP, unet_with_lora


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
            Phix     = x @ Phi.T
            grad     = (Phix - y) @ Phi
            r        = x - self.alphas[k] * grad
            r_spatial = r.view(B, 1, H, W)
            # Residual learning: x_new = r + ΔUNet(r)
            x = (r_spatial + unet_with_lora(self.backbone, self.lora_banks[k],
                                            r_spatial)).view(B, H * W).clamp(0., 1.)

        return x.view(B, 1, H, W)


class LoRADUNCond(nn.Module):
    """Stage 3: condition-dependent LoRA (shared backbone + LoRA + ConditionMLP).

    At each stage k a scalar condition c^(k) is computed from the current
    estimate and mapped by a per-stage MLP to a denoiser scale s^(k) ∈ (0,2):

        x^{k+1} = r^k + s^k · UNet_k(r^k)

    condition_type:
        're'  (Stage 3A) — residual energy  e^(k) = mean(||Φx^k - y||²)
        'mc'  (Stage 3B) — gradient norm²   g^(k) = mean(||Φᵀ(Φx^k-y)||²)

    ConditionMLP is initialized so s ≡ 1 (same as LoRADUNFixed at start).
    """

    def __init__(self, num_stages: int = 10, channels=(32, 64, 128),
                 patch_size: int = 64, rank: int = 4,
                 condition_type: str = 're'):
        super().__init__()
        assert condition_type in ('re', 'mc')
        self.num_stages     = num_stages
        self.patch_size     = patch_size
        self.condition_type = condition_type
        self.backbone  = LightUNet(channels)
        self.lora_banks = nn.ModuleList(
            [LoRABank(channels, rank) for _ in range(num_stages)]
        )
        self.alphas    = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_stages)]
        )
        self.cond_mlps = nn.ModuleList(
            [ConditionMLP() for _ in range(num_stages)]
        )

    def forward(self, y: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        H = W = self.patch_size
        B = y.shape[0]
        x = y @ Phi

        for k in range(self.num_stages):
            Phix     = x @ Phi.T
            residual = Phix - y               # (B, M)
            grad     = residual @ Phi         # (B, N)
            r        = x - self.alphas[k] * grad

            if self.condition_type == 're':
                c_k = (residual ** 2).mean(dim=1)   # residual energy  (B,)
            else:
                c_k = (grad ** 2).mean(dim=1)       # gradient norm²   (B,)

            s_k      = self.cond_mlps[k](c_k).view(B, 1, 1, 1)
            r_spatial = r.view(B, 1, H, W)
            x = (r_spatial + s_k * unet_with_lora(
                    self.backbone, self.lora_banks[k], r_spatial)
                 ).view(B, H * W).clamp(0., 1.)

        return x.view(B, 1, H, W)


class LoRADUNDR(nn.Module):
    """Stage 4: Dynamic Rank LoRA.

    Each unrolled stage gets its own rank from a preset sequence.
    Default: decreasing [8,8,6,6,4,4,2,2,1,1] — early stages need more
    expressivity for coarse reconstruction, later stages need less.
    """

    def __init__(self, num_stages: int = 10, channels=(32, 64, 128),
                 patch_size: int = 64,
                 ranks=(8, 8, 6, 6, 4, 4, 2, 2, 1, 1)):
        super().__init__()
        assert len(ranks) == num_stages, "len(ranks) must equal num_stages"
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.backbone   = LightUNet(channels)
        self.lora_banks = nn.ModuleList(
            [LoRABank(channels, rank=r) for r in ranks]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_stages)]
        )

    def forward(self, y: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        H = W = self.patch_size
        B = y.shape[0]
        x = y @ Phi

        for k in range(self.num_stages):
            Phix      = x @ Phi.T
            grad      = (Phix - y) @ Phi
            r         = x - self.alphas[k] * grad
            r_spatial = r.view(B, 1, H, W)
            x = (r_spatial + unet_with_lora(self.backbone, self.lora_banks[k],
                                            r_spatial)).view(B, H * W).clamp(0., 1.)

        return x.view(B, 1, H, W)


class LoRADUNGate(nn.Module):
    """Stage 5: Gated Dynamic Rank LoRA.

    Per-stage gate g^(k) ∈ (0,1) controls how much LoRA contributes:
        backbone_out = backbone(r^k)
        lora_out     = unet_with_lora(backbone, LoRA_k, r^k)
        output       = backbone_out + g^(k) · (lora_out - backbone_out)

    g^(k) = Sigmoid(GateMLP_k(e^(k))),  e^(k) = mean(||Φx^k - y||²)

    When g→0: stage behaves as Shared-DUN (LoRA suppressed, backbone only).
    When g→1: stage behaves as LoRADUNDR (full LoRA).
    Initialized at g ≈ 1 so training starts identical to Stage 4.

    Gate is applied at the stage output level (not per-layer) to avoid
    training instability from per-sample gating inside conv layers.
    """

    def __init__(self, num_stages: int = 10, channels=(32, 64, 128),
                 patch_size: int = 64,
                 ranks=(1, 1, 2, 2, 4, 4, 6, 6, 8, 8)):
        super().__init__()
        assert len(ranks) == num_stages
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.backbone   = LightUNet(channels)
        self.lora_banks = nn.ModuleList(
            [LoRABank(channels, rank=r) for r in ranks]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(num_stages)]
        )
        self.gate_mlps = nn.ModuleList(
            [GateMLP() for _ in range(num_stages)]
        )

    def forward(self, y: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        H = W = self.patch_size
        B = y.shape[0]
        x = y @ Phi

        for k in range(self.num_stages):
            Phix     = x @ Phi.T
            residual = Phix - y
            grad     = residual @ Phi
            r        = x - self.alphas[k] * grad

            e_k = (residual ** 2).mean(dim=1)          # (B,)
            g_k = self.gate_mlps[k](e_k).view(B, 1, 1, 1)  # (B,1,1,1) ∈ (0,1)

            r_spatial    = r.view(B, 1, H, W)
            backbone_out = self.backbone(r_spatial)
            lora_out     = unet_with_lora(self.backbone, self.lora_banks[k], r_spatial)
            # Interpolate: g=1 → full LoRA, g=0 → backbone only
            out = backbone_out + g_k * (lora_out - backbone_out)

            x = (r_spatial + out).view(B, H * W).clamp(0., 1.)

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
    if name == 'LoRADUNCondRE':
        return LoRADUNCond(**kwargs, rank=cfg.get('lora_rank', 4), condition_type='re')
    if name == 'LoRADUNCondMC':
        return LoRADUNCond(**kwargs, rank=cfg.get('lora_rank', 4), condition_type='mc')
    if name == 'LoRADUNDR':
        ranks = tuple(cfg.get('lora_ranks', [8, 8, 6, 6, 4, 4, 2, 2, 1, 1]))
        return LoRADUNDR(**kwargs, ranks=ranks)
    if name == 'LoRADUNGate':
        ranks = tuple(cfg.get('lora_ranks', [1, 1, 2, 2, 4, 4, 6, 6, 8, 8]))
        return LoRADUNGate(**kwargs, ranks=ranks)
    raise ValueError(f'Unknown model: {name}')
