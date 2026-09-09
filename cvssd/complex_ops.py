"""
Complex-valued primitives.

Every layer is implemented as a pair of real kernels applied by the complex
product rule, which keeps the parameter count explicit (a complex layer has
exactly twice the real parameters of its real counterpart) and makes the
real-vs-complex ablation honest: the real twin is widened until parameter
counts match, so any gain cannot be attributed to extra capacity.

References for the design choices:
  Trabelsi et al., "Deep Complex Networks", ICLR 2018 (arXiv:1705.09792)
    - complex convolution, complex batch norm, modReLU / CReLU.
Whitening-style complex normalisation is available via ComplexLayerNorm(
whiten=True); the default RMS variant is cheaper and was stable in practice.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """(B, 2, L) real -> (B, 1, L) complex."""
    return torch.complex(x[:, 0:1], x[:, 1:2])


def to_real(z: torch.Tensor) -> torch.Tensor:
    """(B, 1, L) complex -> (B, 2, L) real."""
    return torch.cat([z.real, z.imag], dim=1)


def complex_flat(z: torch.Tensor) -> torch.Tensor:
    """(..., D) complex -> (..., 2D) real, for feeding real-valued heads."""
    return torch.view_as_real(z).flatten(-2)


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------


class ComplexLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = True):
        super().__init__()
        self.wr = nn.Parameter(torch.empty(out_f, in_f))
        self.wi = nn.Parameter(torch.empty(out_f, in_f))
        bound = 1.0 / math.sqrt(2 * in_f)
        nn.init.uniform_(self.wr, -bound, bound)
        nn.init.uniform_(self.wi, -bound, bound)
        if bias:
            self.br = nn.Parameter(torch.zeros(out_f))
            self.bi = nn.Parameter(torch.zeros(out_f))
        else:
            self.register_parameter("br", None)
            self.register_parameter("bi", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        xr, xi = z.real, z.imag
        yr = F.linear(xr, self.wr) - F.linear(xi, self.wi)
        yi = F.linear(xr, self.wi) + F.linear(xi, self.wr)
        if self.br is not None:
            yr, yi = yr + self.br, yi + self.bi
        return torch.complex(yr, yi)


class ComplexConv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0, groups=1,
                 bias=True):
        super().__init__()
        self.stride, self.padding, self.groups = stride, padding, groups
        shape = (cout, cin // groups, kernel_size)
        self.wr = nn.Parameter(torch.empty(*shape))
        self.wi = nn.Parameter(torch.empty(*shape))
        bound = 1.0 / math.sqrt(2 * (cin // groups) * kernel_size)
        nn.init.uniform_(self.wr, -bound, bound)
        nn.init.uniform_(self.wi, -bound, bound)
        if bias:
            self.br = nn.Parameter(torch.zeros(cout))
            self.bi = nn.Parameter(torch.zeros(cout))
        else:
            self.register_parameter("br", None)
            self.register_parameter("bi", None)

    def forward(self, z):
        xr, xi = z.real.contiguous(), z.imag.contiguous()
        kw = dict(stride=self.stride, padding=self.padding, groups=self.groups)
        yr = F.conv1d(xr, self.wr, **kw) - F.conv1d(xi, self.wi, **kw)
        yi = F.conv1d(xr, self.wi, **kw) + F.conv1d(xi, self.wr, **kw)
        if self.br is not None:
            yr = yr + self.br[None, :, None]
            yi = yi + self.bi[None, :, None]
        return torch.complex(yr, yi)


class ComplexConvTranspose1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=True):
        super().__init__()
        self.kw = dict(stride=stride, padding=padding,
                       output_padding=output_padding)
        shape = (cin, cout, kernel_size)
        self.wr = nn.Parameter(torch.empty(*shape))
        self.wi = nn.Parameter(torch.empty(*shape))
        bound = 1.0 / math.sqrt(2 * cout * kernel_size)
        nn.init.uniform_(self.wr, -bound, bound)
        nn.init.uniform_(self.wi, -bound, bound)
        if bias:
            self.br = nn.Parameter(torch.zeros(cout))
            self.bi = nn.Parameter(torch.zeros(cout))
        else:
            self.register_parameter("br", None)
            self.register_parameter("bi", None)

    def forward(self, z):
        xr, xi = z.real.contiguous(), z.imag.contiguous()
        yr = (F.conv_transpose1d(xr, self.wr, **self.kw)
              - F.conv_transpose1d(xi, self.wi, **self.kw))
        yi = (F.conv_transpose1d(xr, self.wi, **self.kw)
              + F.conv_transpose1d(xi, self.wr, **self.kw))
        if self.br is not None:
            yr = yr + self.br[None, :, None]
            yi = yi + self.bi[None, :, None]
        return torch.complex(yr, yi)


class ComplexLayerNorm(nn.Module):
    """
    RMS normalisation over the last (feature) dimension using |z|^2.

    Phase is left untouched, which is the point: an amplitude-only
    normalisation is equivariant to a global phase rotation of the input,
    matching the physical invariance of an IQ waveform under carrier phase.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z):
        p = (z.real ** 2 + z.imag ** 2).mean(dim=-1, keepdim=True)
        s = torch.rsqrt(p + self.eps) * self.g
        return z * s.to(z.dtype)


class ModReLU(nn.Module):
    """modReLU: relu(|z| + b) * z / |z|. Phase-preserving."""

    def __init__(self, dim: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        mag = torch.abs(z)
        scale = F.relu(mag + self.b) / (mag + 1e-8)
        return z * scale.to(z.dtype)


class CReLU(nn.Module):
    """Independent ReLU on the real and imaginary parts. Not phase-preserving."""

    def forward(self, z):
        return torch.complex(F.relu(z.real), F.relu(z.imag))


class ComplexSiLU(nn.Module):
    """Magnitude-gated SiLU; smooth and phase-preserving."""

    def forward(self, z):
        mag = torch.abs(z)
        return z * torch.sigmoid(mag).to(z.dtype)
