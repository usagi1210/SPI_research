"""
Mask-Guided SPI Reconstruction Network (MG-SPINet)

Inspired by MST (Mask-guided Spectral-Temporal Transformer) in CASSI,
adapted for single-pixel imaging.

Key idea: the measurement matrix Phi plays the role of the physical mask in CASSI.
Two guidance signals are derived from Phi at each unrolled phase:
  1. Coverage map  (static):  diag(Phi^T Phi) reshaped to 33x33
                              -> which pixels are well-measured by this Phi
  2. Residual projection (dynamic): Phi^T(y - Phi*x) reshaped to 33x33
                              -> where the current estimate is still inaccurate

These two signals are fused into a spatial attention map that modulates
the feature maps inside each unrolled ISTA phase.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MaskAttention(nn.Module):
    """
    Fuse coverage map and residual projection into a channel-wise
    spatial attention that modulates intermediate features.
    """
    def __init__(self, channels: int):
        super().__init__()
        # 2 input channels: coverage + residual_projection
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feat, coverage, residual):
        """
        feat     : (B, C, 33, 33)
        coverage : (B, 1, 33, 33)  normalised diag(Phi^T Phi)
        residual : (B, 1, 33, 33)  normalised Phi^T(y - Phi*x)
        """
        guidance = torch.cat([coverage, residual], dim=1)   # (B, 2, 33, 33)
        attn = self.encoder(guidance)                        # (B, C, 33, 33)
        return feat * attn


class MaskGuidedBlock(nn.Module):
    """
    One unrolled ISTA phase with mask-guided attention.

    Differences from ISTA-Net+ BasicBlock:
    - MaskAttention inserted between encoder and soft-threshold
    - Receives per-phase coverage and residual_projection tensors
    """
    def __init__(self, channels: int = 32):
        super().__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr    = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D         = nn.Parameter(init.xavier_normal_(torch.Tensor(channels,  1, 3, 3)))
        self.conv1_forward  = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels, 3, 3)))
        self.conv2_forward  = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels, 3, 3)))
        self.conv_G         = nn.Parameter(init.xavier_normal_(torch.Tensor(1, channels, 3, 3)))

        self.mask_attn = MaskAttention(channels)

    def forward(self, x, PhiTPhi, PhiTb, coverage, residual):
        """
        x        : (B, 1089)
        PhiTPhi  : (1089, 1089)
        PhiTb    : (B, 1089)
        coverage : (B, 1, 33, 33)
        residual : (B, 1, 33, 33)
        """
        # --- gradient descent step ---
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_2d = x.view(-1, 1, 33, 33)

        # --- encoder ---
        x_D   = F.conv2d(x_2d, self.conv_D, padding=1)
        x_fwd = F.relu(F.conv2d(x_D, self.conv1_forward, padding=1))
        x_fwd = F.conv2d(x_fwd, self.conv2_forward, padding=1)

        # --- mask-guided attention ---
        x_fwd = self.mask_attn(x_fwd, coverage, residual)

        # --- proximal / soft-threshold ---
        x_thr = torch.mul(torch.sign(x_fwd), F.relu(torch.abs(x_fwd) - self.soft_thr))

        # --- decoder ---
        x_bwd  = F.relu(F.conv2d(x_thr, self.conv1_backward, padding=1))
        x_bwd  = F.conv2d(x_bwd, self.conv2_backward, padding=1)
        x_G    = F.conv2d(x_bwd, self.conv_G, padding=1)
        x_pred = (x_2d + x_G).view(-1, 1089)

        # --- symmetric loss term (same as ISTA-Net+) ---
        x_sym   = F.relu(F.conv2d(x_fwd, self.conv1_backward, padding=1))
        x_D_est = F.conv2d(x_sym, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return x_pred, symloss


class MGSPINet(nn.Module):
    """Mask-Guided SPI Reconstruction Network."""

    def __init__(self, num_layers: int = 9, channels: int = 32):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [MaskGuidedBlock(channels) for _ in range(num_layers)]
        )

    def forward(self, Phix, Phi, Qinit):
        """
        Phix  : (B, M)   compressed measurements
        Phi   : (M, N)   measurement matrix
        Qinit : (N, M)   initialisation matrix (learned least-squares)
        """
        B = Phix.shape[0]

        PhiTPhi = torch.mm(Phi.T, Phi)           # (N, N)
        PhiTb   = torch.mm(Phix, Phi)            # (B, N)

        # --- static coverage map: diag(Phi^T Phi) ---
        # high value → pixel well-covered by measurements
        cov_vec = torch.diag(PhiTPhi)             # (N,)
        cov_vec = cov_vec / (cov_vec.max() + 1e-8)
        coverage = cov_vec.view(1, 1, 33, 33).expand(B, -1, -1, -1)  # (B,1,33,33)

        # --- initial estimate ---
        x = torch.mm(Phix, Qinit.T)              # (B, N)

        sym_losses = []
        for layer in self.layers:
            # dynamic residual projection: Phi^T (y - Phi x)
            meas_residual = Phix - torch.mm(x, Phi.T)        # (B, M)
            res_proj      = torch.mm(meas_residual, Phi)      # (B, N)
            res_proj      = res_proj.view(B, 1, 33, 33)
            r_scale       = res_proj.abs().amax(dim=[1, 2, 3], keepdim=True) + 1e-8
            res_proj      = res_proj / r_scale                # normalise per sample

            x, sym = layer(x, PhiTPhi, PhiTb, coverage, res_proj)
            sym_losses.append(sym)

        return x, sym_losses
