import os
import glob
import argparse
import numpy as np
import scipy.io as sio
import cv2
import torch
import torch.nn as nn
from time import time

from model import MGSPINet
from utils import imread_cs, img2col, col2im, compute_psnr, compute_ssim

parser = argparse.ArgumentParser(description='Test MG-SPINet')
parser.add_argument('--cs_ratio',   type=int,   default=25, choices=[1, 4, 10, 25, 30, 40, 50])
parser.add_argument('--num_layers', type=int,   default=9)
parser.add_argument('--channels',   type=int,   default=32)
parser.add_argument('--epoch_num',  type=int,   default=200)
parser.add_argument('--gpu',        type=str,   default='0')
parser.add_argument('--test_set',   type=str,   default='Set11')
parser.add_argument('--data_dir',   type=str,   default='../../data/test')
parser.add_argument('--matrix_dir', type=str,   default='../../matrices')
parser.add_argument('--ckpt_dir',   type=str,   default='../../results/ours/checkpoints')
parser.add_argument('--result_dir', type=str,   default='../../results/ours/images')
parser.add_argument('--log_dir',    type=str,   default='../../results/ours/logs')
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
except Exception:
    pass

# ---------------------------------------------------------------------------
# Load Phi and Qinit
# ---------------------------------------------------------------------------
phi_path   = os.path.join(args.matrix_dir, f'phi_0_{args.cs_ratio}_1089.mat')
Phi_np     = sio.loadmat(phi_path)['phi'].astype(np.float32)
qinit_path = os.path.join(args.matrix_dir, f'Initialization_Matrix_{args.cs_ratio}.mat')
Qinit_np   = sio.loadmat(qinit_path)['Qinit'].astype(np.float32)

Phi   = torch.from_numpy(Phi_np).to(device)
Qinit = torch.from_numpy(Qinit_np).to(device)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
ckpt_path = os.path.join(
    args.ckpt_dir,
    f'mgspi_layer{args.num_layers}_ch{args.channels}_ratio{args.cs_ratio}_epoch{args.epoch_num}.pth'
)
model = MGSPINet(num_layers=args.num_layers, channels=args.channels)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f'Loaded: {ckpt_path}')

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
test_dir  = os.path.join(args.data_dir, args.test_set)
img_paths = sorted(glob.glob(os.path.join(test_dir, '*.tif')) +
                   glob.glob(os.path.join(test_dir, '*.png')) +
                   glob.glob(os.path.join(test_dir, '*.bmp')))

save_dir = os.path.join(args.result_dir, args.test_set, f'ratio_{args.cs_ratio}')
os.makedirs(save_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

psnr_list, ssim_list = [], []
print(f'\nTesting MGSPINet | {args.test_set} | ratio={args.cs_ratio}% | {len(img_paths)} images\n')

with torch.no_grad():
    for idx, img_path in enumerate(img_paths):
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        Iorg_y  = img_yuv[:, :, 0]

        Iorg, row, col, Ipad, row_new, col_new = imread_cs(Iorg_y)
        Icol = img2col(Ipad) / 255.0

        batch = torch.from_numpy(Icol.astype(np.float32)).to(device)
        Phix  = torch.mm(batch, Phi.T)

        t0 = time()
        x_out, _ = model(Phix, Phi, Qinit)
        elapsed  = time() - t0

        pred  = x_out.cpu().numpy()
        X_rec = np.clip(col2im(pred, row, col, row_new, col_new), 0, 1)

        psnr_val = compute_psnr(X_rec * 255, Iorg.astype(np.float64))
        ssim_val = compute_ssim(X_rec * 255, Iorg.astype(np.float64))
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

        print(f'[{idx+1:02d}/{len(img_paths)}] {os.path.basename(img_path):20s} '
              f'PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  t={elapsed:.3f}s')

        img_rec_yuv          = img_yuv.copy()
        img_rec_yuv[:, :, 0] = (X_rec * 255).astype(np.uint8)
        img_rec_bgr          = cv2.cvtColor(img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        base                 = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(
            os.path.join(save_dir, f'{base}_PSNR{psnr_val:.2f}_SSIM{ssim_val:.4f}.png'),
            img_rec_bgr
        )

avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)
summary  = (f'\nAvg PSNR={avg_psnr:.2f}  Avg SSIM={avg_ssim:.4f} '
            f'| set={args.test_set}  ratio={args.cs_ratio}%  epoch={args.epoch_num}\n')
print(summary)

log_path = os.path.join(args.log_dir, f'test_{args.test_set}_ratio{args.cs_ratio}.txt')
with open(log_path, 'a') as f:
    f.write(summary)
