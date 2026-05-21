import os
import platform
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import ISTANetPlus

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train ISTA-Net+')
parser.add_argument('--cs_ratio',       type=int,   default=25,    choices=[1, 4, 10, 25, 30, 40, 50])
parser.add_argument('--num_layers',     type=int,   default=9,     help='number of unrolled phases')
parser.add_argument('--lr',             type=float, default=1e-4)
parser.add_argument('--epochs',         type=int,   default=200)
parser.add_argument('--batch_size',     type=int,   default=64)
parser.add_argument('--gpu',            type=str,   default='0')
parser.add_argument('--data_dir',       type=str,   default='../../data/train/BSD400')
parser.add_argument('--matrix_dir',     type=str,   default='../../matrices')
parser.add_argument('--ckpt_dir',       type=str,   default='../../results/ISTA_Net/checkpoints')
parser.add_argument('--log_dir',        type=str,   default='../../results/ISTA_Net/logs')
parser.add_argument('--resume_epoch',   type=int,   default=0,     help='resume from this epoch (0 = scratch)')
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
# Ratio → measurement dimension mapping
# ---------------------------------------------------------------------------
RATIO_TO_M = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
n_input  = RATIO_TO_M[args.cs_ratio]
n_output = 1089   # 33 × 33

# ---------------------------------------------------------------------------
# Load sampling matrix and training data
# ---------------------------------------------------------------------------
phi_path = os.path.join(args.matrix_dir, f'phi_0_{args.cs_ratio}_1089.mat')
Phi_np   = sio.loadmat(phi_path)['phi'].astype(np.float32)          # (M, 1089)

train_path    = os.path.join(args.data_dir, 'Training_Data.mat')
training_data = sio.loadmat(train_path)
labels        = training_data['labels'].astype(np.float32)           # (N, 1089)
nrtrain       = labels.shape[0]

# ---------------------------------------------------------------------------
# Qinit: least-squares initialisation matrix
# ---------------------------------------------------------------------------
qinit_path = os.path.join(args.matrix_dir, f'Initialization_Matrix_{args.cs_ratio}.mat')
if os.path.exists(qinit_path):
    Qinit_np = sio.loadmat(qinit_path)['Qinit'].astype(np.float32)
else:
    X   = labels.T                            # (1089, N)
    Y   = Phi_np @ X                          # (M, N)
    YYT = Y @ Y.T
    XYT = X @ Y.T
    Qinit_np = (XYT @ np.linalg.inv(YYT)).astype(np.float32)
    os.makedirs(args.matrix_dir, exist_ok=True)
    sio.savemat(qinit_path, {'Qinit': Qinit_np})
    print(f'Saved Qinit to {qinit_path}')

Phi   = torch.from_numpy(Phi_np).to(device)
Qinit = torch.from_numpy(Qinit_np).to(device)

# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


num_workers = 0 if platform.system() == 'Windows' else 4
loader = DataLoader(
    PatchDataset(labels),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = ISTANetPlus(args.num_layers)
model = nn.DataParallel(model).to(device)

os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.log_dir,  exist_ok=True)

ckpt_prefix = os.path.join(
    args.ckpt_dir,
    f'ista_net_plus_layer{args.num_layers}_ratio{args.cs_ratio}'
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
print(f'Training: layers={args.num_layers}, ratio={args.cs_ratio}%, device={device}')

for epoch in range(args.resume_epoch + 1, args.epochs + 1):
    model.train()
    for batch in loader:
        batch = batch.to(device)
        Phix  = torch.mm(batch, Phi.T)

        x_out, sym_losses = model(Phix, Phi, Qinit)

        loss_recon      = torch.mean((x_out - batch) ** 2)
        loss_sym        = sum(torch.mean(s ** 2) for s in sym_losses)
        loss            = loss_recon + gamma * loss_sym

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
