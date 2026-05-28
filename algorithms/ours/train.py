"""Training entry point for LoRA-DUN variants.

Usage:
    python train.py --config configs/base_dun.yaml --cr 0.10
    torchrun --nproc_per_node=2 train.py --config configs/base_dun.yaml --cr 0.10 --distributed
"""
import os
import sys
import glob
import platform
import argparse
import logging
import time
import yaml
import math
import numpy as np
import scipy.io as sio
import cv2
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from lora_dun import build_model
from dataset import BSD400Dataset, load_test_image, img_to_blocks, blocks_to_img
from loss import build_loss

torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """img1, img2 in [0, 1]."""
    mse = np.mean((img1 - img2) ** 2)
    return 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=1.0)


# ---------------------------------------------------------------------------
# Validation + visualisation
# ---------------------------------------------------------------------------

def _get_img_paths(val_dir: str) -> list:
    return sorted(
        glob.glob(os.path.join(val_dir, '*.tif')) +
        glob.glob(os.path.join(val_dir, '*.png')) +
        glob.glob(os.path.join(val_dir, '*.bmp'))
    )


def _reconstruct(model, Phi, img_path: str, patch_size: int,
                 device, batch_size: int = 64):
    """Return (orig_img, rec_img) both float [0,1]."""
    img = load_test_image(img_path)
    blocks, ph, pw, h0, w0 = img_to_blocks(img, patch_size)
    rec_blocks = []
    with torch.no_grad():
        for i in range(0, len(blocks), batch_size):
            chunk = torch.from_numpy(blocks[i:i+batch_size]).to(device)
            out   = model(chunk @ Phi.T, Phi)
            rec_blocks.append(
                out.squeeze(1).view(-1, patch_size * patch_size).cpu().numpy()
            )
    rec_img = blocks_to_img(np.concatenate(rec_blocks, 0), ph, pw, h0, w0, patch_size)
    return img, rec_img


def validate(model, Phi, val_dir: str, patch_size: int,
             device, batch_size: int = 64) -> tuple:
    """Returns (mean_psnr, mean_ssim). Model is set back to train mode after."""
    img_paths = _get_img_paths(val_dir)
    if not img_paths:
        return None, None

    model.eval()
    psnr_list, ssim_list = [], []
    for path in img_paths:
        img, rec = _reconstruct(model, Phi, path, patch_size, device, batch_size)
        psnr_list.append(compute_psnr(rec, img))
        ssim_list.append(compute_ssim(rec, img))
    model.train()
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def save_vis(model, Phi, val_dir: str, patch_size: int,
             device, save_dir: str, epoch: int, batch_size: int = 64):
    """Save gt | rec side-by-side PNGs. Called only on new-best epochs."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for path in _get_img_paths(val_dir):
        name = os.path.splitext(os.path.basename(path))[0]
        img, rec = _reconstruct(model, Phi, path, patch_size, device, batch_size)
        vis = np.concatenate([img, rec], axis=1)
        cv2.imwrite(
            os.path.join(save_dir, f'{name}_ep{epoch:03d}.png'),
            (vis * 255).astype(np.uint8)
        )
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      type=str, required=True)
    parser.add_argument('--cr',          type=float, default=None)
    parser.add_argument('--gpu',         type=str,  default='0')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--resume',      type=str,  default=None)
    parser.add_argument('--run_id',      type=str,  default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.cr is not None:
        cfg['cr'] = args.cr

    cr     = cfg['cr']
    cr_pct = int(round(cr * 100))

    # ---- Distributed setup ------------------------------------------------
    if args.distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device('cuda', local_rank)
        rank   = dist.get_rank()
    else:
        os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        rank   = 0

    is_main = (rank == 0)

    # ---- Directories ------------------------------------------------------
    run_id   = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(cfg['result_dir'], f'cr{cr_pct}', run_id)
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    log_dir  = os.path.join(base_dir, 'logs')
    vis_root = os.path.join(base_dir, 'vis')   # visualisation root
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir,  exist_ok=True)

    # ---- Logger -----------------------------------------------------------
    logger = None
    if is_main:
        log_path = os.path.join(log_dir, f'train_cr{cr_pct}.log')
        logger   = build_logger(log_path)
        logger.info(f'Run ID : {run_id}')
        logger.info(f'Config : {args.config}')
        logger.info(f'Model  : {cfg["model_name"]}  stages={cfg["num_stages"]}  '
                    f'channels={cfg["channels"]}')
        logger.info(f'CR     : {cr_pct}%  epochs={cfg["epochs"]}  '
                    f'batch={cfg["batch_size"]}  lr={cfg["lr"]}')
        logger.info(f'Device : {device}  distributed={args.distributed}')

    # ---- Measurement matrix -----------------------------------------------
    N        = cfg['patch_size'] ** 2
    mat_file = os.path.join(cfg['matrix_dir'], f'phi_{cr}_{N}.mat')
    if not os.path.exists(mat_file):
        raise FileNotFoundError(
            f'Matrix not found: {mat_file}\n'
            f'Run: python ../../utils/gen_matrices.py'
        )
    Phi_np = sio.loadmat(mat_file)['phi'].astype(np.float32)
    Phi    = torch.from_numpy(Phi_np).to(device)

    if is_main:
        logger.info(f'Matrix : {mat_file}  shape={Phi_np.shape}')

    # ---- Dataset ----------------------------------------------------------
    num_workers = 0 if platform.system() == 'Windows' else cfg.get('num_workers', 4)
    dataset = BSD400Dataset(
        cfg['train_dir'],
        patch_size=cfg['patch_size'],
        patches_per_image=cfg.get('patches_per_image', 50),
    )
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        loader  = DataLoader(dataset, batch_size=cfg['batch_size'],
                             sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        loader = DataLoader(dataset, batch_size=cfg['batch_size'],
                            shuffle=True, num_workers=num_workers, pin_memory=True)

    if is_main:
        logger.info(f'Train  : {len(dataset)} patches  {len(loader)} iters/epoch')

    # ---- Model ------------------------------------------------------------
    model = build_model(cfg).to(device)
    if is_main:
        total = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f'Params : {total:.2f} M')

    start_epoch = 0
    best_psnr   = 0.0
    best_path   = os.path.join(ckpt_dir, f'best_cr{cr_pct}.pth')

    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ---- Optimiser + scheduler -------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['epochs'], eta_min=cfg.get('lr_min', 1e-5),
    )

    if args.resume:
        ckpt        = torch.load(args.resume, map_location=device)
        net         = model.module if args.distributed else model
        net.load_state_dict(ckpt['model'])
        start_epoch = ckpt.get('epoch', 0)
        best_psnr   = ckpt.get('best_psnr', 0.0)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if is_main:
            logger.info(f'Resumed from {args.resume} (epoch {start_epoch})')

    # ---- Loss + loop config -----------------------------------------------
    compute_loss = build_loss(cfg)
    iter_step    = cfg.get('iter_step', 100)
    val_every    = cfg.get('val_every', 1)    # default: test every epoch
    save_freq    = cfg.get('save_freq', 10)

    if is_main:
        logger.info('Training started.\n')

    # ---- Training loop ----------------------------------------------------
    for epoch in range(start_epoch + 1, cfg['epochs'] + 1):
        if args.distributed:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for it, batch in enumerate(loader):
            batch  = batch.to(device)               # (B, 1, p, p)
            B      = batch.shape[0]
            x_flat = batch.view(B, -1)              # (B, N)
            y      = x_flat @ Phi.T                 # (B, M)

            pred = model(y, Phi)                    # (B, 1, p, p)
            loss = compute_loss(pred, batch, y, Phi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if is_main and it % iter_step == 0:
                logger.info(
                    f'epoch {epoch:<3d}  iter {it:<4d}  '
                    f'loss {loss.item():.5f}  lr {optimizer.param_groups[0]["lr"]:.6f}'
                )

        scheduler.step()

        if is_main:
            elapsed  = time.time() - t0
            avg_loss = epoch_loss / (it + 1)
            msg = (f'epoch {epoch:<3d}  avg_loss {avg_loss:.5f}  '
                   f'lr {optimizer.param_groups[0]["lr"]:.6f}  time {elapsed:.1f}s')

            # ---- Validation ------------------------------------------------
            if epoch % val_every == 0:
                net = model.module if args.distributed else model
                psnr, ssim = validate(net, Phi, cfg['val_dir'],
                                      cfg['patch_size'], device)
                if psnr is not None:
                    msg += f'  psnr {psnr:.2f} dB  ssim {ssim:.4f}'
                    if psnr > best_psnr:
                        best_psnr = psnr
                        torch.save(
                            {'epoch': epoch, 'model': net.state_dict(),
                             'best_psnr': best_psnr},
                            best_path
                        )
                        # Visualise only on new-best epochs (overwrites vis/best/)
                        save_vis(net, Phi, cfg['val_dir'], cfg['patch_size'],
                                 device, os.path.join(vis_root, 'best'), epoch)
                        msg += '  [best]'

            # ---- Periodic named checkpoint ---------------------------------
            if epoch % save_freq == 0:
                net = model.module if args.distributed else model
                torch.save(
                    {'epoch': epoch, 'model': net.state_dict(),
                     'best_psnr': best_psnr},
                    os.path.join(ckpt_dir, f'epoch{epoch}_cr{cr_pct}.pth')
                )

            # ---- latest.pth (overwrite every epoch for crash recovery) -----
            net = model.module if args.distributed else model
            torch.save(
                {'epoch': epoch, 'model': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'best_psnr': best_psnr},
                os.path.join(ckpt_dir, f'latest_cr{cr_pct}.pth')
            )

            logger.info(msg + '\n')

    if is_main:
        logger.info(f'Training finished. Best val PSNR: {best_psnr:.2f} dB')
        logger.info(f'Best checkpoint : {best_path}')

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
