'''
Code for denoising images
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import datasets

import torchvision
import torchvision.transforms as transforms

import scipy.io as mlio
import numpy as np

import os
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import math
from einops import rearrange, repeat

from src.models.nn import DropoutNd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

# Define the path to store the CelebA dataset
data_dir = './data/CelebA'

res1 = 2048
res2 = 128

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.CenterCrop(178),  # Center crop to 178x178
    transforms.Resize((res1,res2)),      # Resize to 128x128
    transforms.ToTensor(),       # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    transforms.Lambda(lambda x: x.view(3, res1*res2)),
])

# Load the CelebA dataset
dataset = datasets.CelebA(root=data_dir, split='train', transform=transform, download=False)

# Create DataLoader for the dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Check the dataset
print(f'Number of samples: {len(dataset)}')


"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

class S4DKernel_simple(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=1, lr=0.0000):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(1e-4) - math.log(1e-4)
        ) + math.log(1e-4)

        #alpha = 490.52 # Flip
        alpha = math.pi # Raw
        #alpha = math.pi / 3 # Enhance
        #alpha = math.pi * 10 # Middle
        shift = 0
        A_imag = repeat(torch.arange(N // 2), 'n -> h n', h=H) * alpha + shift
        log_A_real = torch.log(3 * torch.ones(H, N//2))
        A_imag[:,0] *= 0
        log_A_real[:,0] = -5

        C = torch.randn(H, N // 2, dtype=torch.cfloat) * torch.abs(A_imag) / 1000
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, 0)

        self.register("log_A_real", log_A_real, 0)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

L_image = res1*res2
class S4D_simple(nn.Module):
    def __init__(self, d_state = 128, L = L_image, d_output = 3, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.n = d_state
        self.d_output = d_output
        self.d_model = 3
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(1))
        self.encoder = nn.Linear(3, self.d_model)
        self.decoder = nn.Linear(self.d_model, d_output)

        # SSM Kernel
        self.kernel = S4DKernel_simple(self.d_model, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k = nn.functional.pad(k,(0,L),'constant',0)
        u = nn.functional.pad(u,(0,L),'constant',0)
        k_f = torch.fft.fft(k) # (H L)
        u_f = torch.fft.fft(u) # (B H L)

        #beta = -1
        #beta = -0.5
        beta = 0
        #beta = 0.5
        #beta = 1
        scales = repeat((1 + torch.arange(L, device=k.device) / 10000) ** beta, 'l -> h l', h=self.d_model)
        scales_flipped = torch.flip(scales, dims=[1])
        scales = torch.cat((scales, scales_flipped), dim=1)
        k_f = k_f * scales

        y = torch.fft.ifft(u_f*k_f).real # (B H L)
        y = y[...,0:L]
        u = u[...,0:L]

        y = self.output_linear(y)

        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified


model = S4D_simple()
model = model.to('cuda')

epochs = 20
criterion = nn.MSELoss()

all_parameters = list(model.parameters())
params = [p for p in all_parameters if not hasattr(p, "_optim")]
optimizer = optim.AdamW(params, lr=0.001, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs+1)

# Add parameters with special hyperparameters
hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
hps = [
    dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
]  # Unique dicts
for hp in hps:
    params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
    optimizer.add_param_group(
        {"params": params, **hp}
    )

keys = sorted(set([k for hp in hps for k in hp.keys()]))
for i, g in enumerate(optimizer.param_groups):
    group_hps = {k: g.get(k, None) for k in keys}
    print(' | '.join([
        f"Optimizer group {i}",
        f"{len(g['params'])} tensors",
    ] + [f"{k} {v}" for k, v in group_hps.items()]))

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train():
    model.train()
    train_loss = 0
    total = 0
    pbar = tqdm(enumerate(dataloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.4f' %
            (batch_idx, len(dataloader), train_loss/(batch_idx+1))
        )
    return loss

loss = -1

if __name__ == '__main__':
    pbar = tqdm(range(start_epoch, epochs))
    for epoch in pbar:
        if epoch == 0:
            pbar.set_description('Epoch: %d' % (epoch))
        else:
            pbar.set_description('Epoch: %d | Loss: %1.4f' % (epoch, loss))
        loss = train()
        scheduler.step()

    state = {
        'model': model.state_dict(),
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')