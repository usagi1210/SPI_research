import os
import glob
import platform
import argparse
from datetime import datetime
import numpy as np
import scipy.io as sio
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import MGSPINet
from utils import imread_cs, img2col, col2im, compute_psnr

parser = argparse.ArgumentParser(description='Train MG-SPINet')
parser.add_argument('--cs_ratio',     type=int,   default=25, choices=[1, 4, 10, 25, 30, 40, 50])
parser.add_argument('--num_layers',   type=int,   default=9)
parser.add_argument('--channels',     type=int,   default=32)
parser.add_argument('--lr',           type=float, default=1e-4)
parser.add_argument('--epochs',       type=int,   default=200)
parser.add_argument('--batch_size',   type=int,   default=64)
parser.add_argument('--gpu',          type=str,   default='0')
parser.add_argument('--data_dir',     type=str,   default='../../data/train/BSD400')
parser.add_argument('--matrix_dir',   type=str,   default='../../matrices')
parser.add_argument('--result_dir',   type=str,   default='../../results/ours')
parser.add_argument('--run_id',       type=str,   default='', help='run identifier (default: auto timestamp)')
parser.add_argument('--resume_epoch', type=int,   default=0)
parser.add_argument('--resume_run',   type=str,   default='')
parser.add_argument('--val_dir',      type=str,   default='../../data/test/Set11')
parser.add_argument('--val_every',    type=int,   default=10)
args = parser.parse_args()

run_id   = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir  = os.path.join(args.result_dir, run_id)
ckpt_dir = os.path.join(run_dir, 'checkpoints')
log_dir  = os.path.join(run_dir, 'logs')
print(f'Run ID: {run_id}  ->  {run_dir}')

os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False
except Exception:
    pass

RATIO_TO_M = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
phi_path = os.path.join(args.matrix_dir, f'phi_0_{args.cs_ratio}_1089.mat')
Phi_np   = sio.loadmat(phi_path)['phi'].astype(np.float32)

train_path = os.path.join(args.data_dir, 'Training_Data.mat')
labels     = sio.loadmat(train_path)['labels'].astype(np.float32)
nrtrain    = labels.shape[0]

qinit_path = os.path.join(args.matrix_dir, f'Initialization_Matrix_{args.cs_ratio}.mat')
if os.path.exists(qinit_path):
    Qinit_np = sio.loadmat(qinit_path)['Qinit'].astype(np.float32)
else:
    X        = labels.T
    Y        = Phi_np @ X
    Qinit_np = (X @ Y.T @ np.linalg.inv(Y @ Y.T)).astype(np.float32)
    sio.savemat(qinit_path, {'Qinit': Qinit_np})

Phi   = torch.from_numpy(Phi_np).to(device)
Qinit = torch.from_numpy(Qinit_np).to(device)

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

num_workers = 0 if platform.system() == 'Windows' else 4
loader = DataLoader(PatchDataset(labels), batch_size=args.batch_size,
                    shuffle=True, num_workers=num_workers, pin_memory=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = MGSPINet(num_layers=args.num_layers, channels=args.channels)
model = nn.DataParallel(model).to(device)

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(log_dir,  exist_ok=True)

ckpt_prefix = os.path.join(ckpt_dir, f'mgspi_layer{args.num_layers}_ch{args.channels}_ratio{args.cs_ratio}')
log_path    = os.path.join(log_dir,  f'train_layer{args.num_layers}_ratio{args.cs_ratio}.txt')

if args.resume_epoch > 0:
    resume_run = args.resume_run or run_id
    resume_ckpt_dir = os.path.join(args.result_dir, resume_run, 'checkpoints')
    ckpt = os.path.join(resume_ckpt_dir, f'mgspi_layer{args.num_layers}_ch{args.channels}_ratio{args.cs_ratio}_epoch{args.resume_epoch}.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f'Resumed from {ckpt}')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
gamma     = torch.tensor(0.01, device=device)
best_psnr = 0.0
best_path = f'{ckpt_prefix}_best.pth'

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------
def validate():
    img_paths = sorted(
        glob.glob(os.path.join(args.val_dir, '*.tif')) +
        glob.glob(os.path.join(args.val_dir, '*.png')) +
        glob.glob(os.path.join(args.val_dir, '*.bmp'))
    )
    if not img_paths:
        return None
    psnr_list = []
    model.eval()
    with torch.no_grad():
        for path in img_paths:
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            Iorg_y  = img_yuv[:, :, 0]
            Iorg, row, col, Ipad, row_new, col_new = imread_cs(Iorg_y)
            Icol  = img2col(Ipad) / 255.0
            batch = torch.from_numpy(Icol.astype(np.float32)).to(device)
            Phix  = torch.mm(batch, Phi.T)
            x_out, _ = model(Phix, Phi, Qinit)
            pred  = x_out.cpu().numpy()
            X_rec = np.clip(col2im(pred, row, col, row_new, col_new), 0, 1)
            psnr_list.append(compute_psnr(X_rec * 255, Iorg.astype(np.float64)))
    model.train()
    return float(np.mean(psnr_list))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f'Training MGSPINet: layers={args.num_layers}, ch={args.channels}, '
      f'ratio={args.cs_ratio}%, device={device}')

for epoch in range(args.resume_epoch + 1, args.epochs + 1):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        Phix  = torch.mm(batch, Phi.T)

        x_out, sym_losses = model(Phix, Phi, Qinit)

        loss_recon = torch.mean((x_out - batch) ** 2)
        loss_sym   = sum(torch.mean(s ** 2) for s in sym_losses)
        loss       = loss_recon + gamma * loss_sym

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log_line = (f'[{epoch:03d}/{args.epochs}] '
                f'total={loss.item():.4f}  recon={loss_recon.item():.4f}  '
                f'sym={loss_sym.item():.4f}')

    if epoch % args.val_every == 0:
        val_psnr = validate()
        if val_psnr is not None:
            log_line += f'  val_psnr={val_psnr:.2f}'
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), best_path)
                log_line += '  [best]'

    log_line += '\n'
    print(log_line, end='')
    with open(log_path, 'a') as f:
        f.write(log_line)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'{ckpt_prefix}_epoch{epoch}.pth')

print(f'Training finished. Best val PSNR: {best_psnr:.2f} dB  ->  {best_path}')
