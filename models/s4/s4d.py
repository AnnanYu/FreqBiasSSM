"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from src.models.nn import DropoutNd

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=4, lr=0.001, sf = 1):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(1e-4) - math.log(1e-4)
        ) + math.log(1e-4)

        C = torch.randn(H, N, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, 0)

        log_A_real = torch.log(0.5 * torch.ones(H, N))
        A_imag = math.pi * repeat(1.3 + torch.arange(N), 'n -> h n', h=H) * sf
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, 0.1)

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


class S4D(nn.Module):
    def __init__(self, d_state=4, L = 62832, d_output = 4, dropout=0.0, alpha = 1, beta = 1, transposed=True, **kernel_args):
        super().__init__()

        self.n = d_state
        self.d_output = d_output
        self.d_model = 16
        self.transposed = transposed
        self.D = nn.Parameter(torch.randn(1))
        self.encoder = nn.Linear(1, self.d_model)
        self.decoder = nn.Linear(self.d_model, 4)
        self.alpha = alpha
        self.beta = beta

        # SSM Kernel
        self.kernel = S4DKernel(self.d_model, N=self.n, sf=self.alpha, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.GELU(),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        u = torch.unsqueeze(u,1) # (B 1 L)
        u = u.transpose(-1,-2) # (B L 1)
        u = self.encoder(u) # (B L H)
        u = u.transpose(-1,-2) # (B H L)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.fft(k) # (H L)
        u_f = torch.fft.fft(u) # (B H L)
        
        sob_filter = (1 + torch.arange(L, device=u_f.device).view(1, 1, -1) / 10000) ** self.beta # (L)
        y = torch.fft.ifft(k_f * u_f * sob_filter).real # (B H L)
        # u = u[...,0:L]
        y = y + self.D * u

        y = self.output_linear(y)
        
        y = y.transpose(-1,-2) # (B L H)
        y = self.decoder(y) # (B 4)
        y = y.mean(dim=1) # (B H)

        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified
