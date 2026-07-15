import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.dw = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.dw(x)
        return self.act(self.norm(x))


class LiteFullProxNet(nn.Module):
    """Light full-image proximal network for 256x256 SPI reconstruction."""

    def __init__(self, channels=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.enc1 = SepConvBlock(1, c1)
        self.enc2 = SepConvBlock(c1, c2)
        self.enc3 = SepConvBlock(c2, c3)
        self.bot = nn.Sequential(
            SepConvBlock(c3, c3),
            SepConvBlock(c3, c3),
        )
        self.dec3 = SepConvBlock(c3 + c3, c2)
        self.dec2 = SepConvBlock(c2 + c2, c1)
        self.dec1 = nn.Conv2d(c1 + c1, 1, 3, padding=1)
        nn.init.zeros_(self.dec1.weight)
        nn.init.zeros_(self.dec1.bias)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))

        d = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        return self.dec1(torch.cat([d, e1], dim=1))


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.GELU(),
        )


class FullConvProxNet(nn.Module):
    """Full-image version of the current LightUNet-style proximal network."""

    def __init__(self, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.enc1 = ConvBlock(1, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bot = ConvBlock(c3, c3)
        self.dec3 = ConvBlock(c3 + c3, c2)
        self.dec2 = ConvBlock(c2 + c2, c1)
        self.dec1 = nn.Conv2d(c1 + c1, 1, 3, padding=1)
        nn.init.zeros_(self.dec1.weight)
        nn.init.zeros_(self.dec1.bias)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))

        d = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False)
        return self.dec1(torch.cat([d, e1], dim=1))


class FullSharedDUN(nn.Module):
    """Full-image Kronecker sensing DUN.

    Measurement model:
        Y = H X W^T

    One unfolding stage:
        R_k = X_k - alpha_k * H^T(H X_k W^T - Y)W
        X_{k+1} = R_k + shared_prox(R_k)
    """

    def __init__(
        self,
        image_size: int = 256,
        meas_size: int = 81,
        num_stages: int = 7,
        channels=(16, 32, 64),
        prox_type: str = "conv",
        matrix_train: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.meas_size = meas_size
        self.num_stages = num_stages

        self.H = nn.Parameter(torch.empty(meas_size, image_size), requires_grad=matrix_train)
        self.W = nn.Parameter(torch.empty(meas_size, image_size), requires_grad=matrix_train)
        nn.init.xavier_normal_(self.H)
        nn.init.xavier_normal_(self.W)

        if prox_type == "conv":
            self.prox = FullConvProxNet(channels)
        elif prox_type == "sep":
            self.prox = LiteFullProxNet(channels)
        else:
            raise ValueError(f"Unknown prox_type: {prox_type}")
        self.alphas = nn.Parameter(torch.full((num_stages,), 0.5))

    def measure(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mh,bchw,nw->bcmn", self.H, x, self.W)

    def backproject(self, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mh,bcmn,nw->bchw", self.H, y, self.W)

    def reconstruct(self, y: torch.Tensor) -> torch.Tensor:
        x = self.backproject(y)
        for k in range(self.num_stages):
            residual = self.measure(x) - y
            grad = self.backproject(residual)
            alpha = F.softplus(self.alphas[k])
            r = x - alpha * grad
            x = (r + self.prox(r)).clamp(0.0, 1.0)
        return x

    def forward(self, x_gt: torch.Tensor) -> torch.Tensor:
        y = self.measure(x_gt)
        return self.reconstruct(y)


def build_full_model(cfg: dict) -> nn.Module:
    name = cfg.get("model_name", "FullSharedDUN")
    kwargs = dict(
        image_size=cfg.get("image_size", 256),
        meas_size=cfg.get("meas_size", 81),
        num_stages=cfg.get("num_stages", 7),
        channels=tuple(cfg.get("channels", [16, 32, 64])),
        prox_type=cfg.get("prox_type", "conv"),
        matrix_train=cfg.get("matrix_train", True),
    )
    if name == "FullSharedDUN":
        return FullSharedDUN(**kwargs)
    raise ValueError(f"Unknown full-image model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def compression_ratio(image_size: int, meas_size: int) -> float:
    return (meas_size * meas_size) / float(image_size * image_size)
