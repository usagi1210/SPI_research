import torch
import torch.nn as nn


class ISTAStep(nn.Module):
    """One unrolled ISTA stage: gradient step + learnable denoiser (proximal op)."""

    def __init__(self, denoiser: nn.Module):
        super().__init__()
        self.denoiser = denoiser
        self.alpha = nn.Parameter(torch.tensor(0.1))   # learnable step size

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                Phi: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        x   : (B, N)  current estimate (vectorised)
        y   : (B, M)  measurements
        Phi : (M, N)  measurement matrix
        Returns (B, N) updated estimate.
        """
        # Gradient of 0.5 * ||Phi x - y||^2 w.r.t. x
        Phix = x @ Phi.T            # (B, M)
        grad = (Phix - y) @ Phi     # (B, N) = Phi^T (Phi x - y)
        r = x - self.alpha * grad   # gradient descent

        # Proximal operator via U-Net
        B = x.shape[0]
        x_new = self.denoiser(r.view(B, 1, H, W)).view(B, H * W)
        return x_new
