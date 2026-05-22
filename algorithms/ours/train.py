import os
import glob
import platform
import argparse
import logging
import time
from datetime import datetime
import numpy as np
import scipy.io as sio
import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import MGSPINet
from utils import imread_cs, img2col, col2im, compute_psnr

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train MG-SPINet')
parser.add_argument('--cs_ratio',     type=int,   default=25, choices=[1, 4, 10, 25, 30, 40, 50])
parser.add_argument('--num_layers',   type=int,   default=9)
parser.add_argument('--channels',     type=int,   default=32)
parser.add_argument('--lr',           type=float, default=1e-4)
parser.add_argument('--epochs',       type=int,   default=200)
parser.add_argument('--batch_size',   type=int,   default=64)
parser.add_argument('--gpu',          type=str,   default='0')
parser.add_argument('--distributed',  action='store_true')
parser.add_argument('--num_workers',  type=int,   default=4)
parser.add_argument('--data_dir',     type=str,   default='../../data/train/BSD400')
parser.add_argument('--matrix_dir',   type=str,   default='../../matrices')
parser.add_argument('--result_dir',   type=str,   default='../../results/ours')
parser.add_argument('--run_id',       type=str,   default='')
parser.add_argument('--resume_epoch', type=int,   default=0)
parser.add_argument('--resume_run',   type=str,   default='')
parser.add_argument('--val_dir',      type=str,   default='../../data/test/Set11')
parser.add_argument('--val_every',    type=int,   default=10)
parser.add_argument('--iter_step',    type=int,   default=100)
args = parser.parse_args()

torch.set_float32_matmul_precision('highest')

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
if args.distributed:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)
    rank = dist.get_rank()
else:
    os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rank = 0

is_main = (rank == 0)

# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------
run_id   = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir  = os.path.join(args.result_dir, run_id)
ckpt_dir = os.path.join(run_dir, 'checkpoints')
log_dir  = os.path.join(run_dir, 'logs')

if is_main:
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def build_logger(log_path):
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

logger = None
if is_main:
    log_path = os.path.join(log_dir, f'train_layer{args.num_layers}_ratio{args.cs_ratio}.txt')
    logger = build_logger(log_path)
    logger.info(f'Run ID : {run_id}')
    logger.info(f'Run dir: {run_dir}')
    logger.info(f'layers={args.num_layers}  ch={args.channels}  ratio={args.cs_ratio}%  '
                f'lr={args.lr}  epochs={args.epochs}  batch={args.batch_size}  '
                f'distributed={args.distributed}')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
phi_path = os.path.join(args.matrix_dir, f'phi_0_{args.cs_ratio}_1089.mat')
Phi_np   = sio.loadmat(phi_path)['phi'].astype(np.float32)

labels = sio.loadmat(os.path.join(args.data_dir, 'Training_Data.mat'))['labels'].astype(np.float32)

qinit_path = os.path.join(args.matrix_dir, f'Initialization_Matrix_{args.cs_ratio}.mat')
if os.path.exists(qinit_path):
    Qinit_np = sio.loadmat(qinit_path)['Qinit'].astype(np.float32)
else:
    X        = labels.T
    Y        = Phi_np @ X
    Qinit_np = (X @ Y.T @ np.linalg.inv(Y @ Y.T)).astype(np.float32)
    if is_main:
        sio.savemat(qinit_path, {'Qinit': Qinit_np})
        logger.info(f'Saved Qinit -> {qinit_path}')

Phi   = torch.from_numpy(Phi_np).to(device)
Qinit = torch.from_numpy(Qinit_np).to(device)

class PatchDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = PatchDataset(labels)
num_workers = 0 if platform.system() == 'Windows' else args.num_workers
if args.distributed:
    sampler = DistributedSampler(dataset, shuffle=True)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         sampler=sampler, num_workers=num_workers, pin_memory=True)
else:
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = MGSPINet(num_layers=args.num_layers, channels=args.channels).to(device)

ckpt_prefix = os.path.join(ckpt_dir, f'mgspi_layer{args.num_layers}_ch{args.channels}_ratio{args.cs_ratio}')
best_path   = f'{ckpt_prefix}_best.pth'

if args.resume_epoch > 0:
    resume_run = args.resume_run or run_id
    src = os.path.join(args.result_dir, resume_run, 'checkpoints',
                       f'mgspi_layer{args.num_layers}_ch{args.channels}_ratio{args.cs_ratio}_epoch{args.resume_epoch}.pth')
    model.load_state_dict(torch.load(src, map_location=device))
    if is_main:
        logger.info(f'Resumed from {src}')

if args.distributed:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
gamma     = torch.tensor(0.01, device=device)
best_psnr = 0.0

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate():
    img_paths = sorted(
        glob.glob(os.path.join(args.val_dir, '*.tif')) +
        glob.glob(os.path.join(args.val_dir, '*.png')) +
        glob.glob(os.path.join(args.val_dir, '*.bmp'))
    )
    if not img_paths:
        return None
    net = model.module if args.distributed else model
    net.eval()
    psnr_list = []
    with torch.no_grad():
        for path in img_paths:
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            Iorg_y  = img_yuv[:, :, 0]
            Iorg, row, col, Ipad, row_new, col_new = imread_cs(Iorg_y)
            Icol  = img2col(Ipad) / 255.0
            batch = torch.from_numpy(Icol.astype(np.float32)).to(device)
            Phix  = torch.mm(batch, Phi.T)
            x_out, _ = net(Phix, Phi, Qinit)
            pred  = x_out.cpu().numpy()
            X_rec = np.clip(col2im(pred, row, col, row_new, col_new), 0, 1)
            psnr_list.append(compute_psnr(X_rec * 255, Iorg.astype(np.float64)))
    net.train()
    return float(np.mean(psnr_list))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if is_main:
    logger.info('Training started.')

for epoch in range(args.resume_epoch + 1, args.epochs + 1):
    if args.distributed:
        sampler.set_epoch(epoch)

    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for iteration, batch in enumerate(loader):
        batch = batch.to(device)
        Phix  = torch.mm(batch, Phi.T)

        x_out, sym_losses = model(Phix, Phi, Qinit)

        loss_recon = torch.mean((x_out - batch) ** 2)
        loss_sym   = sum(torch.mean(s ** 2) for s in sym_losses)
        loss       = loss_recon + gamma * loss_sym

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if is_main and iteration % args.iter_step == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'epoch {epoch:<3d}  iter {iteration:<4d}  '
                f'loss {loss.item():.5f}  recon {loss_recon.item():.5f}  '
                f'sym {loss_sym.item():.5f}  lr {lr:.6f}'
            )

    if is_main:
        elapsed = time.time() - t0
        avg_loss = epoch_loss / (iteration + 1)
        lr = optimizer.param_groups[0]['lr']
        msg = (f'epoch {epoch:<3d}  avg_loss {avg_loss:.5f}  '
               f'lr {lr:.6f}  time {elapsed:.1f}s')

        if epoch % args.val_every == 0:
            val_psnr = validate()
            if val_psnr is not None:
                msg += f'  val_psnr {val_psnr:.2f} dB'
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    net_to_save = model.module if args.distributed else model
                    torch.save(net_to_save.state_dict(), best_path)
                    msg += '  [best]'
        logger.info(msg + '\n')

        if epoch % 5 == 0:
            net_to_save = model.module if args.distributed else model
            torch.save(net_to_save.state_dict(), f'{ckpt_prefix}_epoch{epoch}.pth')

if is_main:
    logger.info(f'Training finished. Best val PSNR: {best_psnr:.2f} dB')

if args.distributed:
    dist.destroy_process_group()
