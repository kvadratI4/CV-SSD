"""Dataset wrapper and the composite training objective."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _c2r(z: np.ndarray) -> np.ndarray:
    return np.stack([z.real, z.imag], axis=0).astype(np.float32)


class IQDenoiseDataset(Dataset):
    """Serves (noisy, clean, snr_norm, index) from a datagen .npz file."""

    def __init__(self, path: str, indices=None, snr_scale: float = 20.0):
        d = np.load(path, allow_pickle=True)
        self.f = {k: d[k] for k in d.files}
        n = len(self.f["noisy"])
        self.idx = np.arange(n) if indices is None else np.asarray(indices)
        self.snr_scale = snr_scale

    def __len__(self):
        return len(self.idx)

    def meta(self, i):
        j = self.idx[i]
        f = self.f
        return dict(
            mod=str(f["mods"][f["mod_id"][j]]), snr_db=float(f["snr_db"][j]),
            cfo=float(f["cfo"][j]), channel=f["channel"][j],
            symbols=f["symbols"][j], sym_idx=f["sym_idx"][j].astype(np.int64),
            sps=int(f["sps"]), n_sym=int(f["n_sym"]), group=int(f["group"][j]),
        )

    def __getitem__(self, i):
        j = self.idx[i]
        return (
            torch.from_numpy(_c2r(self.f["noisy"][j])),
            torch.from_numpy(_c2r(self.f["clean"][j])),
            torch.tensor([self.f["snr_db"][j] / self.snr_scale], dtype=torch.float32),
            int(j),
        )


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------


def waveform_loss(est, clean):
    """Complex MSE, normalised per example so all SNR cells weigh equally."""
    num = ((est - clean) ** 2).sum(dim=(1, 2))
    den = (clean ** 2).sum(dim=(1, 2)) + 1e-8
    return (num / den).mean()


def phase_loss(est, clean, eps: float = 1e-6):
    """
    Magnitude-weighted phase consistency.

    Plain MSE is dominated by amplitude; demodulation is dominated by phase.
    Weighting by the reference magnitude keeps low-energy samples from
    injecting noise into the gradient.
    """
    ec = torch.complex(est[:, 0], est[:, 1])
    cc = torch.complex(clean[:, 0], clean[:, 1])
    w = torch.abs(cc)
    cos = (ec * cc.conj()).real / (torch.abs(ec) * w + eps)
    return ((1 - cos) * w).sum() / (w.sum() + eps)


def spectral_loss(est, clean, n_fft: int = 256, hop: int = 64):
    """L1 on the STFT magnitude, to stop the denoiser flattening the spectrum."""
    win = torch.hann_window(n_fft, device=est.device)
    ec = torch.complex(est[:, 0], est[:, 1])
    cc = torch.complex(clean[:, 0], clean[:, 1])
    kw = dict(n_fft=n_fft, hop_length=hop, window=win, return_complex=True,
              center=True)
    return F.l1_loss(torch.stft(ec, **kw).abs(), torch.stft(cc, **kw).abs())


class CompositeLoss(torch.nn.Module):
    def __init__(self, w_phase=0.1, w_spec=0.1, w_snr=0.05):
        super().__init__()
        self.w_phase, self.w_spec, self.w_snr = w_phase, w_spec, w_snr

    def forward(self, est, clean, snr_hat=None, snr_true=None):
        parts = {"wave": waveform_loss(est, clean)}
        if self.w_phase:
            parts["phase"] = self.w_phase * phase_loss(est, clean)
        if self.w_spec:
            parts["spec"] = self.w_spec * spectral_loss(est, clean)
        if self.w_snr and snr_hat is not None and snr_true is not None:
            parts["snr"] = self.w_snr * F.mse_loss(snr_hat, snr_true)
        return sum(parts.values()), {k: float(v) for k, v in parts.items()}
