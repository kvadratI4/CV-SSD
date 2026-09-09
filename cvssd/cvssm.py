"""
Selective state-space layers, complex and real.

Core recurrence (identical in both cases, only the dtype differs):

    h_t = Abar_t * h_{t-1} + Bbar_t * x_t ,      y_t = <C_t, h_t> + D * x_t

with Abar_t = exp(Delta_t * A) and A diagonal. In the complex layer A is
complex diagonal, exactly as in S4/S4D, where the normal-plus-low-rank state
matrix is unitarily conjugated into complex diagonal form; Mamba-3 likewise
re-introduces complex state. This is the argument that CV-SSD is a natural
extension rather than an arbitrary pairing: the SSM state was already complex,
and the signal it processes is natively complex, so keeping the whole path in C
removes an artificial real/imaginary split rather than adding machinery.

The scan is a chunked Hillis-Steele associative scan written in pure PyTorch.
It is dtype-agnostic and therefore shared by both layers. Note that the official
mamba-ssm CUDA kernel cannot be substituted here: it does not support complex
input. Bounded memory comes from the chunking, not from a fused kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvssd.complex_ops import ComplexLinear, complex_flat, ComplexLayerNorm, ComplexSiLU

# --------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------


def _hillis_steele(a: torch.Tensor, b: torch.Tensor):
    """
    Inclusive scan of h_t = a_t h_{t-1} + b_t over dim=-3, assuming h_{-1}=0.

    Shapes: a, b are (..., C, D, N). Returns (cumprod_a, h).
    """
    C = a.shape[-3]
    step = 1
    while step < C:
        a_sh = torch.cat(
            [torch.ones_like(a[..., :step, :, :]), a[..., : C - step, :, :]],
            dim=-3,
        )
        b_sh = torch.cat(
            [torch.zeros_like(b[..., :step, :, :]), b[..., : C - step, :, :]],
            dim=-3,
        )
        b = b + a * b_sh
        a = a * a_sh
        step *= 2
    return a, b


def selective_scan(a: torch.Tensor, b: torch.Tensor, chunk: int = 64):
    """
    Chunked associative scan. a, b: (B, L, D, N). Returns h: (B, L, D, N).

    Peak memory is O(B * chunk * D * N) instead of O(B * L * D * N * log L),
    which is what makes long sequences trainable without a fused kernel.
    """
    B, L, D, N = a.shape
    pad = (-L) % chunk
    if pad:
        # Explicit concatenation rather than F.pad: constant padding of complex
        # tensors is not reliably supported across PyTorch versions. Identity
        # elements of the scan monoid are a=1, b=0, so padding is a no-op.
        shape = (B, pad, D, N)
        a = torch.cat([a, torch.ones(shape, dtype=a.dtype, device=a.device)], 1)
        b = torch.cat([b, torch.zeros(shape, dtype=b.dtype, device=b.device)], 1)
    Lp = a.shape[1]
    n_chunk = Lp // chunk
    a = a.reshape(B, n_chunk, chunk, D, N)
    b = b.reshape(B, n_chunk, chunk, D, N)

    carry = torch.zeros(B, D, N, dtype=b.dtype, device=b.device)
    outs = []
    for c in range(n_chunk):
        cp, h = _hillis_steele(a[:, c], b[:, c])
        h = h + cp * carry.unsqueeze(1)
        carry = h[:, -1]
        outs.append(h)
    h = torch.cat(outs, dim=1).reshape(B, Lp, D, N)
    return h[:, :L]


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------


def _dt_bias(d_model: int, dt_min: float = 1e-3, dt_max: float = 1e-1):
    dt = torch.exp(
        torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=1e-4)
    return dt + torch.log(-torch.expm1(-dt))          # inverse softplus


class CVSelectiveSSM(nn.Module):
    """
    Complex selective SSM. Input and output: (B, L, D) complex.

    Phase equivariance
    ------------------
    With selection="mag" the step size and the input/output projections B_t, C_t
    are computed from |z| alone. Since the recurrence is complex-linear in z, the
    whole layer then satisfies

        f(exp(i*theta) * z) = exp(i*theta) * f(z)   exactly, for any theta.

    That is the right inductive bias for IQ data: absolute carrier phase is
    nuisance information, so the selection mechanism should not be able to key on
    it. Mamba's usual selection, which reads the real and imaginary parts
    separately, destroys this property - set selection="riparts" to ablate it.
    """

    def __init__(self, d_model: int, d_state: int = 8, chunk: int = 64,
                 selection: str = "mag"):
        super().__init__()
        self.d_model, self.d_state, self.chunk = d_model, d_state, chunk
        self.selection = selection
        # A = -exp(log_re) + i*im  ->  Re(A) < 0  ->  |exp(dt*A)| < 1 always.
        self.A_log_re = nn.Parameter(torch.zeros(d_model, d_state))
        self.A_im = nn.Parameter(
            math.pi * torch.arange(d_state).float().repeat(d_model, 1)
        )
        self.D = nn.Parameter(torch.ones(d_model))
        n_feat = 2 * d_model
        self.dt_proj = nn.Linear(n_feat, d_model)
        self.dt_proj.bias.data = _dt_bias(d_model)
        self.BC_proj = nn.Linear(n_feat, 4 * d_state)

    def A(self):
        return torch.complex(-torch.exp(self.A_log_re), self.A_im)

    def features(self, z):
        if self.selection == "mag":
            m = torch.abs(z)
            return torch.cat([m, torch.log1p(m)], dim=-1)      # phase-invariant
        return complex_flat(z)                                 # (B, L, 2D)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B_, L, D = z.shape
        f = self.features(z)                                  # (B, L, 2D) real
        dt = F.softplus(self.dt_proj(f))                      # (B, L, D)
        bc = self.BC_proj(f).reshape(B_, L, 2, 2, self.d_state)
        Bt = torch.complex(bc[:, :, 0, 0], bc[:, :, 0, 1])    # (B, L, N)
        Ct = torch.complex(bc[:, :, 1, 0], bc[:, :, 1, 1])

        A = self.A()[None, None]                              # (1,1,D,N)
        dtc = dt.unsqueeze(-1).to(z.dtype)                    # (B,L,D,1)
        Abar = torch.exp(dtc * A)
        Bbar = dtc * Bt.unsqueeze(2) * z.unsqueeze(-1)        # (B,L,D,N)

        h = selective_scan(Abar, Bbar, chunk=self.chunk)
        y = (h * Ct.unsqueeze(2)).sum(-1)                     # (B,L,D)
        return y + self.D.to(z.dtype) * z


class RVSelectiveSSM(nn.Module):
    """Real selective SSM (Mamba-style), used for the parameter-matched twin."""

    def __init__(self, d_model: int, d_state: int = 8, chunk: int = 64):
        super().__init__()
        self.d_model, self.d_state, self.chunk = d_model, d_state, chunk
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(d_model, 1)
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.dt_proj = nn.Linear(d_model, d_model)
        self.dt_proj.bias.data = _dt_bias(d_model)
        self.BC_proj = nn.Linear(d_model, 2 * d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, L, D = x.shape
        dt = F.softplus(self.dt_proj(x))
        bc = self.BC_proj(x)
        Bt, Ct = bc[..., : self.d_state], bc[..., self.d_state :]
        A = -torch.exp(self.A_log)[None, None]
        dtc = dt.unsqueeze(-1)
        Abar = torch.exp(dtc * A)
        Bbar = dtc * Bt.unsqueeze(2) * x.unsqueeze(-1)
        h = selective_scan(Abar, Bbar, chunk=self.chunk)
        y = (h * Ct.unsqueeze(2)).sum(-1)
        return y + self.D * x


class FiLM(nn.Module):
    """
    Feature-wise modulation from the estimated SNR.

    Blind at inference: the conditioning scalar comes from the SNR head, never
    from an oracle. In complex mode the modulation is a real gain only - adding a
    complex offset would rotate the constellation by a learned constant and break
    the phase equivariance of the surrounding layers.
    """

    def __init__(self, d_model: int, complex_mode: bool = True, d_cond: int = 32):
        super().__init__()
        self.complex_mode = complex_mode
        out = d_model if complex_mode else 2 * d_model
        self.net = nn.Sequential(
            nn.Linear(1, d_cond), nn.SiLU(), nn.Linear(d_cond, out)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.d_model = d_model

    def forward(self, z, cond):
        p = self.net(cond)                                    # (B, out)
        if self.complex_mode:
            g = (1 + p).unsqueeze(1)
            return z * g.to(z.dtype)
        g, b = p.chunk(2, dim=-1)
        return z * (1 + g).unsqueeze(1) + b.unsqueeze(1)


class SSMBlock(nn.Module):
    """
    Pre-norm SSM block with a gated branch, FiLM conditioning and residual.
    In complex mode the gate is deliberately *real* and computed from |h|. A
    complex gate would multiply two phase-equivariant quantities and produce
    exp(2i*theta) instead of exp(i*theta), silently destroying the equivariance
    that the rest of the block is built to preserve. A magnitude-derived real
    gate modulates amplitude only and leaves phase untouched.
    """

    def __init__(self, d_model, d_state=8, complex_mode=True, chunk=64,
                selection: str = "mag"):
        super().__init__()
        self.complex_mode = complex_mode
        if complex_mode:
            from .complex_ops import ComplexLayerNorm

            self.norm = ComplexLayerNorm(d_model)
            self.ssm = CVSelectiveSSM(d_model, d_state, chunk, selection)
            # real gate on magnitude features; bias-free complex output proj
            self.gate = nn.Linear(d_model, d_model)
            self.out = ComplexLinear(d_model, d_model, bias=False)
            self.act = nn.SiLU()
        else:
            self.norm = nn.LayerNorm(d_model)
            self.ssm = RVSelectiveSSM(d_model, d_state, chunk)
            self.gate = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
            self.act = nn.SiLU()
        self.film = FiLM(d_model, complex_mode)

    def forward(self, z, cond):
        h = self.norm(z)
        h = self.film(h, cond)
        if self.complex_mode:
            g = self.act(self.gate(torch.abs(h)))          # real, phase-invariant
            h = self.ssm(h) * g.to(h.dtype)
        else:
            h = self.ssm(h) * self.act(self.gate(h))
        return z + self.out(h)
