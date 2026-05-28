import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):
    def __init__(self, mode: str = 'l1'):
        super().__init__()
        self.mode = mode

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == 'l1':
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)


class ConsistencyLoss(nn.Module):
    """Physical measurement consistency: ||Phi x_hat - y||^2."""

    def forward(self, pred: torch.Tensor, y: torch.Tensor,
                Phi: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]
        x_flat = pred.view(B, -1)          # (B, N)
        Phix = x_flat @ Phi.T              # (B, M)
        return F.mse_loss(Phix, y)


def build_loss(cfg: dict):
    mode = cfg.get('loss', 'l1')
    recon_w = cfg.get('recon_weight', 1.0)
    cons_w  = cfg.get('consistency_weight', 0.0)   # 0 until Stage 6

    recon_fn = ReconLoss(mode)
    cons_fn  = ConsistencyLoss() if cons_w > 0 else None

    def compute(pred, target, y=None, Phi=None):
        loss = recon_w * recon_fn(pred, target)
        if cons_fn is not None and y is not None and Phi is not None:
            loss = loss + cons_w * cons_fn(pred, y, Phi)
        return loss

    return compute
