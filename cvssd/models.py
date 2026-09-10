"""
Models.

  SSDNet(complex_mode=True)   -> CV-SSD, the proposed complex-valued selective
                                 state-space denoiser.
  SSDNet(complex_mode=False)  -> the real-valued twin, structurally identical.
  Baselines                   -> DnCNN1D, UNet1D, ConvDAE, BiLSTMDenoiser.

All models take a real (B, 2, L) IQ tensor and return (estimate, snr_hat).
They predict the *noise* and subtract it (residual learning, as in DnCNN),
which is consistently easier to optimise than predicting the clean waveform.

The single most important experiment in this repository is
SSDNet(True) vs SSDNet(False) at matched parameter count. Use
`matched_real_width` so that the comparison isolates the complex algebra rather
than capacity; without that control the ablation proves nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cvssd.complex_ops import (ComplexConv1d, ComplexConvTranspose1d, to_complex,
                          to_real)
from cvssd.cvssm import SSMBlock


# --------------------------------------------------------------------------
# blind SNR estimation head
# --------------------------------------------------------------------------


class SNRHead(nn.Module):
    """
    Estimates normalised SNR from the noisy input alone.

    Trained with an auxiliary regression loss against the true SNR, but never
    given it at inference. Reviewers will look for exactly this: many published
    denoisers quietly consume an oracle noise level.

    It reads |x| and log(1+|x|) rather than the raw I/Q pair, so the estimate is
    invariant to a global phase rotation. Classical moment-based SNR estimators
    (M2M4) use exactly these magnitude statistics, and the invariance is what
    lets the conditioned network stay phase-equivariant end to end.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden, 9, stride=4, padding=4), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 9, stride=4, padding=4), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 9, stride=4, padding=4), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        m = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-12).unsqueeze(1)
        h = self.net(torch.cat([m, torch.log1p(m)], dim=1))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        return self.head(h)                                    # (B, 1)


# --------------------------------------------------------------------------
# proposed model
# --------------------------------------------------------------------------


class SSDNet(nn.Module):
    def __init__(
        self,
        d_model: int = 48,
        d_state: int = 8,
        n_stages: int = 3,
        blocks_per_stage: int = 2,
        complex_mode: bool = True,
        chunk: int = 64,
        widths=None,
        selection: str = "mag",
        wl: str = "none",
    ):
        super().__init__()
        self.complex_mode = complex_mode
        self.n_stages = n_stages
        w = widths or [d_model * (2 ** i) for i in range(n_stages + 1)]
        self.widths = w

        if complex_mode:
            # bias=False everywhere: a constant offset added anywhere in the
            # complex path would destroy exact phase equivariance.
            def conv(a, b, k, **kw): return ComplexConv1d(a, b, k, bias=False, **kw)
            def convT(a, b, k, **kw):
                return ComplexConvTranspose1d(a, b, k, bias=False, **kw)
            cin, cout = 1, 1
        else:
            conv, convT, cin, cout = nn.Conv1d, nn.ConvTranspose1d, 2, 2

        self.stem = conv(cin, w[0], 7, padding=3)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(n_stages):
            self.enc.append(nn.ModuleList([
                SSMBlock(w[i], d_state, complex_mode, chunk, selection, wl)
                for _ in range(blocks_per_stage)
            ]))
            self.down.append(conv(w[i], w[i + 1], 4, stride=2, padding=1))

        self.mid = nn.ModuleList([
            SSMBlock(w[n_stages], d_state, complex_mode, chunk, selection, wl)
            for _ in range(blocks_per_stage)
        ])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(n_stages)):
            self.up.append(convT(w[i + 1], w[i], 4, stride=2, padding=1))
            self.dec.append(nn.ModuleList([
                SSMBlock(w[i], d_state, complex_mode, chunk, selection, wl)
                for _ in range(blocks_per_stage)
            ]))
        # Additive, learnable skip weights instead of concatenation: keeps the
        # decoder width fixed and makes the encoder contribution inspectable.
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for _ in range(n_stages)]
        )
        self.head = conv(w[0], cout, 7, padding=3)
        self.snr_head = SNRHead()

    def _blocks(self, mods, z, cond):
        z = z.transpose(1, 2)                                  # (B, L, C)
        for m in mods:
            z = m(z, cond)
        return z.transpose(1, 2)                               # (B, C, L)

    def forward(self, x):
        snr_hat = self.snr_head(x)
        cond = snr_hat.detach() if self.training else snr_hat

        z = to_complex(x) if self.complex_mode else x
        z = self.stem(z)

        skips = []
        for i in range(self.n_stages):
            z = self._blocks(self.enc[i], z, cond)
            skips.append(z)
            z = self.down[i](z)

        z = self._blocks(self.mid, z, cond)

        for j, i in enumerate(reversed(range(self.n_stages))):
            z = self.up[j](z)
            s = skips[i]
            if z.shape[-1] != s.shape[-1]:
                z = z[..., : s.shape[-1]]
            a = self.alpha[i]
            z = z + (a.to(z.dtype) if self.complex_mode else a) * s
            z = self._blocks(self.dec[j], z, cond)

        noise = self.head(z)
        noise = to_real(noise) if self.complex_mode else noise
        return x - noise, snr_hat


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def matched_real_width(d_complex: int, tol: float = 0.02, **kw) -> int:
    """
    Smallest real width whose parameter count matches the complex model.

    Complex layers hold two real kernels, so the real twin needs roughly
    sqrt(2) times the width. Search rather than assume, because the SSM
    projections and the SNR head do not scale identically.
    """
    target = count_params(SSDNet(d_model=d_complex, complex_mode=True, **kw))
    best, best_err = d_complex, float("inf")
    for d in range(d_complex, int(d_complex * 2.5) + 1):
        n = count_params(SSDNet(d_model=d, complex_mode=False, **kw))
        err = abs(n - target) / target
        if err < best_err:
            best, best_err = d, err
        if err < tol:
            break
    return best


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


class DnCNN1D(nn.Module):
    """1D DnCNN: residual noise prediction, the standard denoising baseline."""

    def __init__(self, ch: int = 64, depth: int = 12):
        super().__init__()
        layers = [nn.Conv1d(2, ch, 9, padding=4), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv1d(ch, ch, 9, padding=4, bias=False),
                       nn.BatchNorm1d(ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv1d(ch, 2, 9, padding=4)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.net(x), torch.zeros(x.shape[0], 1, device=x.device)


class _CB(nn.Sequential):
    def __init__(self, cin, cout, k=9, s=1):
        super().__init__(nn.Conv1d(cin, cout, k, s, k // 2),
                         nn.BatchNorm1d(cout), nn.SiLU())


class UNet1D(nn.Module):
    def __init__(self, ch: int = 48, n_stages: int = 3):
        super().__init__()
        w = [ch * (2 ** i) for i in range(n_stages + 1)]
        self.stem = _CB(2, w[0])
        self.enc = nn.ModuleList([_CB(w[i], w[i]) for i in range(n_stages)])
        self.down = nn.ModuleList(
            [_CB(w[i], w[i + 1], 4, 2) for i in range(n_stages)])
        self.mid = _CB(w[-1], w[-1])
        self.up = nn.ModuleList([
            nn.ConvTranspose1d(w[i + 1], w[i], 4, 2, 1)
            for i in reversed(range(n_stages))])
        self.dec = nn.ModuleList([
            _CB(2 * w[i], w[i]) for i in reversed(range(n_stages))])
        self.head = nn.Conv1d(w[0], 2, 9, padding=4)
        self.n_stages = n_stages

    def forward(self, x):
        z = self.stem(x)
        skips = []
        for i in range(self.n_stages):
            z = self.enc[i](z)
            skips.append(z)
            z = self.down[i](z)
        z = self.mid(z)
        for j, i in enumerate(reversed(range(self.n_stages))):
            z = self.up[j](z)
            s = skips[i]
            z = torch.cat([z[..., : s.shape[-1]], s], dim=1)
            z = self.dec[j](z)
        return x - self.head(z), torch.zeros(x.shape[0], 1, device=x.device)


class ConvDAE(nn.Module):
    """Convolutional denoising autoencoder without skips (DRdA-CA family)."""

    def __init__(self, ch: int = 64):
        super().__init__()
        self.enc = nn.Sequential(_CB(2, ch, 9, 2), _CB(ch, ch * 2, 9, 2),
                                 _CB(ch * 2, ch * 2, 9, 2))
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(ch * 2, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose1d(ch * 2, ch, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose1d(ch, 2, 4, 2, 1))

    def forward(self, x):
        y = self.dec(self.enc(x))
        return y[..., : x.shape[-1]], torch.zeros(x.shape[0], 1, device=x.device)


class BiLSTMDenoiser(nn.Module):
    """Residual BiLSTM/BiGRU-style recurrent denoiser (ResBiLSTM-BiGRU family)."""

    def __init__(self, hidden: int = 96, layers: int = 2):
        super().__init__()
        self.inp = nn.Conv1d(2, hidden, 9, padding=4)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(2 * hidden, hidden, 1, batch_first=True,
                          bidirectional=True)
        self.head = nn.Conv1d(2 * hidden, 2, 9, padding=4)

    def forward(self, x):
        h = self.inp(x).transpose(1, 2)
        h, _ = self.lstm(h)
        h, _ = self.gru(h)
        return x - self.head(h.transpose(1, 2)), torch.zeros(
            x.shape[0], 1, device=x.device)


MODELS = {
    "cvssd": lambda **kw: SSDNet(complex_mode=True, **kw),
    "rvssd": lambda **kw: SSDNet(complex_mode=False, **kw),
    # widely-linear variants: the mechanism test for the properness finding
    "cvssd_wl": lambda **kw: SSDNet(complex_mode=True, wl="plain", **kw),
    "cvssd_wleq": lambda **kw: SSDNet(complex_mode=True, wl="eq", **kw),
    "dncnn": lambda **kw: DnCNN1D(),
    "unet": lambda **kw: UNet1D(),
    "dae": lambda **kw: ConvDAE(),
    "bilstm": lambda **kw: BiLSTMDenoiser(),
}
