"""Train full-image Kronecker DUN variants.

Example:
    python train_full.py --config configs/full_shared_dun.yaml --gpu 0
    torchrun --nproc_per_node=4 train_full.py --config configs/full_shared_dun.yaml --distributed
"""
import argparse
import glob
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))
from dataset import BSD400Dataset, load_test_image
from full_model import build_full_model, compression_ratio

torch.set_float32_matmul_precision("highest")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"train_full:{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    ch = logging.StreamHandler()
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=1.0)


def image_to_tiles(img: np.ndarray, tile: int = 256):
    h0, w0 = img.shape
    ph = math.ceil(h0 / tile) * tile
    pw = math.ceil(w0 / tile) * tile
    pad = np.zeros((ph, pw), dtype=np.float32)
    pad[:h0, :w0] = img
    tiles = []
    for r in range(0, ph, tile):
        for c in range(0, pw, tile):
            tiles.append(pad[r:r + tile, c:c + tile])
    return np.stack(tiles, axis=0), ph, pw, h0, w0


def tiles_to_image(tiles: np.ndarray, ph: int, pw: int, h0: int, w0: int, tile: int = 256):
    out = np.zeros((ph, pw), dtype=np.float32)
    idx = 0
    for r in range(0, ph, tile):
        for c in range(0, pw, tile):
            out[r:r + tile, c:c + tile] = tiles[idx]
            idx += 1
    return np.clip(out[:h0, :w0], 0.0, 1.0)


def validate(model, val_dir: str, image_size: int, device, save_dir: str = None):
    paths = sorted(
        glob.glob(os.path.join(val_dir, "*.tif")) +
        glob.glob(os.path.join(val_dir, "*.png")) +
        glob.glob(os.path.join(val_dir, "*.bmp"))
    )
    if not paths:
        return None, None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            img = load_test_image(path)
            tiles, ph, pw, h0, w0 = image_to_tiles(img, image_size)
            batch = torch.from_numpy(tiles[:, None]).to(device)
            rec = model(batch).squeeze(1).cpu().numpy()
            rec_img = tiles_to_image(rec, ph, pw, h0, w0, image_size)
            psnrs.append(compute_psnr(rec_img, img))
            ssims.append(compute_ssim(rec_img, img))
            if save_dir:
                vis = np.concatenate([img, rec_img], axis=1)
                cv2.imwrite(os.path.join(save_dir, f"{name}.png"), (vis * 255).astype(np.uint8))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = int(cfg.get("image_size", 256))
    meas_size = int(cfg.get("meas_size", 81))
    cr_pct = int(round(100 * compression_ratio(image_size, meas_size)))

    if args.distributed:
        backend = "nccl" if torch.cuda.is_available() and platform.system() != "Windows" else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        rank = dist.get_rank()
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rank = 0
        local_rank = 0

    is_main = rank == 0
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(cfg["result_dir"], f"cr{cr_pct}", run_id)
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    vis_root = os.path.join(base_dir, "vis")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    logger = None
    if is_main:
        logger = build_logger(os.path.join(log_dir, f"train_full_cr{cr_pct}.log"))
        logger.info(f"Run ID : {run_id}")
        logger.info(f"Config : {args.config}")
        logger.info(f"Model  : {cfg['model_name']} stages={cfg['num_stages']} channels={cfg['channels']}")
        logger.info(f"Size   : image={image_size} meas={meas_size} CR={compression_ratio(image_size, meas_size):.4f}")
        logger.info(f"Device : {device} distributed={args.distributed}")

    num_workers = 0 if platform.system() == "Windows" else cfg.get("num_workers", 4)
    dataset = BSD400Dataset(
        cfg["train_dir"],
        patch_size=image_size,
        patches_per_image=cfg.get("patches_per_image", 20),
    )
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    model = build_full_model(cfg).to(device)
    if args.distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    net = model.module if args.distributed else model
    if is_main:
        total = sum(p.numel() for p in net.parameters()) / 1e6
        logger.info(f"Params : {total:.3f} M")
        logger.info(f"Train  : {len(dataset)} crops  {len(loader)} iters/epoch")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg.get("lr_min", 1e-5)
    )
    start_epoch = 0
    best_psnr = 0.0
    best_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        best_psnr = ckpt.get("best_psnr", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        if is_main:
            logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    loss_mode = cfg.get("loss", "l1")
    consistency_weight = float(cfg.get("consistency_weight", 0.0))
    val_every = int(cfg.get("val_every", 1))
    save_freq = int(cfg.get("save_freq", 10))
    iter_step = int(cfg.get("iter_step", 50))
    best_path = os.path.join(ckpt_dir, f"best_cr{cr_pct}.pth")

    if is_main:
        logger.info("Training started.\n")

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for it, batch in enumerate(loader):
            batch = batch.to(device)
            pred = model(batch)
            if loss_mode == "mse":
                loss = torch.mean((pred - batch) ** 2)
            else:
                loss = torch.mean(torch.abs(pred - batch))

            if consistency_weight > 0:
                net = model.module if args.distributed else model
                loss = loss + consistency_weight * torch.mean((net.measure(pred) - net.measure(batch)) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if is_main and it % iter_step == 0:
                logger.info(
                    f"epoch {epoch:<3d} iter {it:<4d} loss {loss.item():.5f} "
                    f"lr {optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()

        if is_main:
            avg_loss = epoch_loss / (it + 1)
            msg = (
                f"epoch {epoch:<3d} avg_loss {avg_loss:.5f} "
                f"lr {optimizer.param_groups[0]['lr']:.6f} time {time.time() - t0:.1f}s"
            )

            if epoch % val_every == 0:
                psnr, ssim = validate(
                    model.module if args.distributed else model,
                    cfg["val_dir"],
                    image_size,
                    device,
                    save_dir=os.path.join(vis_root, f"epoch_{epoch:03d}"),
                )
                if psnr is not None:
                    msg += f" psnr {psnr:.2f} dB ssim {ssim:.4f}"
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_epoch = epoch
                        torch.save(
                            {"epoch": epoch, "model": net.state_dict(), "best_psnr": best_psnr, "best_epoch": best_epoch},
                            best_path,
                        )
                        msg += " [best]"

            if epoch % save_freq == 0:
                torch.save(
                    {"epoch": epoch, "model": net.state_dict(), "best_psnr": best_psnr, "best_epoch": best_epoch},
                    os.path.join(ckpt_dir, f"epoch{epoch}_cr{cr_pct}.pth"),
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_psnr": best_psnr,
                    "best_epoch": best_epoch,
                },
                os.path.join(ckpt_dir, f"latest_cr{cr_pct}.pth"),
            )
            logger.info(msg + "\n")

    if is_main:
        logger.info(f"Training finished. Best val PSNR: {best_psnr:.2f} dB (epoch {best_epoch})")
        logger.info(f"Best checkpoint : {best_path}")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
