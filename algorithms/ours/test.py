"""Testing entry point for LoRA-DUN variants.

Usage:
    python test.py --config configs/base_dun.yaml --cr 0.10 \\
                   --ckpt ../../results/ours/base_dun/cr10/<run_id>/checkpoints/best_cr10.pth

    # Also evaluate on BSD68:
    python test.py --config configs/base_dun.yaml --cr 0.10 --ckpt <path> --bsd68
"""
import os
import sys
import glob
import argparse
import math
import yaml
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.dirname(__file__))
from lora_dun import build_model
from dataset import load_test_image, img_to_blocks, blocks_to_img

torch.set_float32_matmul_precision('highest')


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    return 100.0 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, data_range=1.0)


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate(model, Phi, img_dir: str, patch_size: int,
             device, save_dir: str = None, batch_size: int = 64) -> dict:
    img_paths = sorted(
        glob.glob(os.path.join(img_dir, '*.tif')) +
        glob.glob(os.path.join(img_dir, '*.png')) +
        glob.glob(os.path.join(img_dir, '*.bmp'))
    )
    if not img_paths:
        print(f'  No images found in {img_dir}')
        return {}

    results = {}
    model.eval()
    with torch.no_grad():
        for path in img_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            img  = load_test_image(path)
            blocks, ph, pw, h0, w0 = img_to_blocks(img, patch_size)

            rec_blocks = []
            for i in range(0, len(blocks), batch_size):
                chunk = torch.from_numpy(blocks[i:i+batch_size]).to(device)
                y     = chunk @ Phi.T
                out   = model(y, Phi)
                rec_blocks.append(
                    out.squeeze(1).view(-1, patch_size * patch_size).cpu().numpy()
                )

            rec_blocks = np.concatenate(rec_blocks, axis=0)
            rec_img    = blocks_to_img(rec_blocks, ph, pw, h0, w0, patch_size)

            psnr = compute_psnr(rec_img, img)
            ssim = compute_ssim(rec_img, img)
            results[name] = {'psnr': psnr, 'ssim': ssim}

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{name}_rec.png')
                import cv2
                cv2.imwrite(save_path, (rec_img * 255).astype(np.uint8))

    return results


def print_results(title: str, results: dict):
    if not results:
        return
    line = '─' * 52
    print(f'\n{line}')
    print(f'  {title}')
    print(line)
    print(f'  {"Image":<20}  {"PSNR":>7}  {"SSIM":>7}')
    print(line)
    psnrs, ssims = [], []
    for name, v in results.items():
        print(f'  {name:<20}  {v["psnr"]:>7.2f}  {v["ssim"]:>7.4f}')
        psnrs.append(v['psnr'])
        ssims.append(v['ssim'])
    print(line)
    print(f'  {"Average":<20}  {np.mean(psnrs):>7.2f}  {np.mean(ssims):>7.4f}')
    print(f'{line}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--cr',     type=float, default=None)
    parser.add_argument('--ckpt',   type=str,   required=True)
    parser.add_argument('--gpu',    type=str,   default='0')
    parser.add_argument('--bsd68',  action='store_true',
                        help='Also evaluate on BSD68')
    parser.add_argument('--save',   action='store_true',
                        help='Save reconstructed images alongside checkpoint')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.cr is not None:
        cfg['cr'] = args.cr

    cr     = cfg['cr']
    cr_pct = int(round(cr * 100))

    os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ---- Matrix -----------------------------------------------------------
    N        = cfg['patch_size'] ** 2
    mat_file = os.path.join(cfg['matrix_dir'], f'phi_{cr}_{N}.mat')
    Phi_np   = sio.loadmat(mat_file)['phi'].astype(np.float32)
    Phi      = torch.from_numpy(Phi_np).to(device)

    # ---- Model ------------------------------------------------------------
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f'\nModel  : {cfg["model_name"]}  ({total_params:.2f} M params)')
    print(f'CR     : {cr_pct}%')
    print(f'Ckpt   : {args.ckpt}')

    # ---- Evaluate Set11 --------------------------------------------------
    save_dir = None
    if args.save:
        ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
        save_dir  = os.path.join(os.path.dirname(args.ckpt),
                                 f'rec_set11_{ckpt_name}')
    results = evaluate(model, Phi, cfg['val_dir'],
                       cfg['patch_size'], device, save_dir)
    print_results(f'Set11  (CR={cr_pct}%)', results)

    # ---- Evaluate BSD68 --------------------------------------------------
    if args.bsd68:
        bsd68_dir = os.path.join(os.path.dirname(cfg['val_dir']), 'BSD68')
        if not os.path.isdir(bsd68_dir):
            bsd68_dir = cfg.get('bsd68_dir', bsd68_dir)
        if args.save:
            ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
            save_dir  = os.path.join(os.path.dirname(args.ckpt),
                                     f'rec_bsd68_{ckpt_name}')
        res68 = evaluate(model, Phi, bsd68_dir,
                         cfg['patch_size'], device, save_dir)
        print_results(f'BSD68  (CR={cr_pct}%)', res68)


if __name__ == '__main__':
    main()
