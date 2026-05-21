import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr   = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D          = nn.Parameter(init.xavier_normal_(torch.Tensor(32,  1, 3, 3)))
        self.conv1_forward   = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward   = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward  = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward  = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G          = nn.Parameter(init.xavier_normal_(torch.Tensor( 1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        # gradient step
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

        # encoder
        x_D       = F.conv2d(x_input, self.conv_D, padding=1)
        x_fwd     = F.relu(F.conv2d(x_D, self.conv1_forward, padding=1))
        x_forward = F.conv2d(x_fwd, self.conv2_forward, padding=1)

        # soft threshold (proximal operator)
        x_thr = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        # decoder
        x_bwd      = F.relu(F.conv2d(x_thr, self.conv1_backward, padding=1))
        x_backward = F.conv2d(x_bwd, self.conv2_backward, padding=1)
        x_G        = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = (x_input + x_G).view(-1, 1089)

        # symmetric loss term: encoder(decoder(x)) ≈ x
        x_sym_bwd = F.relu(F.conv2d(x_forward, self.conv1_backward, padding=1))
        x_D_est   = F.conv2d(x_sym_bwd, self.conv2_backward, padding=1)
        symloss   = x_D_est - x_D

        return x_pred, symloss


class ISTANetPlus(nn.Module):
    def __init__(self, num_layers: int):
        super(ISTANetPlus, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([BasicBlock() for _ in range(num_layers)])

    def forward(self, Phix, Phi, Qinit):
        PhiTPhi = torch.mm(Phi.T, Phi)
        PhiTb   = torch.mm(Phix, Phi)
        x       = torch.mm(Phix, Qinit.T)

        sym_losses = []
        for layer in self.layers:
            x, sym = layer(x, PhiTPhi, PhiTb)
            sym_losses.append(sym)

        return x, sym_losses
