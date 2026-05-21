import os
import platform
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import MGSPINet

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
parser.add_argument('--ckpt_dir',     type=str,   default='../../results/ours/checkpoints')
parser.add_argument('--log_dir',      type=str,   default='../../results/ours/logs')
parser.add_argument('--resume_epoch', type=int,   default=0)
args = parser.parse_args()

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

os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.log_dir,  exist_ok=True)

ckpt_prefix = os.path.join(
    args.ckpt_dir,
    f'mgspi_layer{args.num_layers}_ch{args.channels}_ratio{args.cs_ratio}'
)
log_path = os.path.join(
    args.log_dir,
    f'train_layer{args.num_layers}_ratio{args.cs_ratio}.txt'
)

if args.resume_epoch > 0:
    ckpt = f'{ckpt_prefix}_epoch{args.resume_epoch}.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f'Resumed from {ckpt}')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
gamma     = torch.tensor(0.01, device=device)

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
                f'sym={loss_sym.item():.4f}\n')
    print(log_line, end='')
    with open(log_path, 'a') as f:
        f.write(log_line)

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'{ckpt_prefix}_epoch{epoch}.pth')

print('Training finished.')
