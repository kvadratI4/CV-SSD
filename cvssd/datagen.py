"""
Paired (clean, noisy) IQ dataset generation for supervised RF denoising.

Ground-truth convention
-----------------------
The clean target is the *noise-free received* waveform, i.e. the transmit
waveform after pulse shaping, multipath, CFO, phase noise and PA nonlinearity,
but before AWGN:

    y_noisy = y_clean + n,     n ~ CN(0, sigma^2)

Rationale: the task under study is denoising, not blind equalisation. Using the
ideal transmit waveform as target would silently fold equalisation into the
metric and make comparison against CNN/GAN denoising baselines unfair.
Set clean_mode="tx" to switch to the ideal-transmit target for ablations.

Every sample carries the transmitted symbols and the receiver-side parameters
needed to compute EVM and BER downstream, plus group ids for leakage-controlled
splitting.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Constellations with Gray mapping
# --------------------------------------------------------------------------

MODULATIONS = ("BPSK", "QPSK", "8PSK", "PAM4", "16QAM", "64QAM")


def _gray(n: int) -> np.ndarray:
    """Binary-reflected Gray code sequence of length n (n a power of two)."""
    i = np.arange(n)
    return i ^ (i >> 1)


def _pam_gray(m: int) -> np.ndarray:
    """m-PAM levels ordered so that index i carries Gray-coded bits."""
    levels = np.arange(m) * 2 - (m - 1)          # ..., -3, -1, 1, 3, ...
    order = np.argsort(_gray(m))                  # gray index -> natural index
    return levels[order].astype(np.float64)


def constellation(mod: str) -> np.ndarray:
    """Unit-average-power constellation, index = Gray-coded symbol value."""
    mod = mod.upper()
    if mod == "BPSK":
        pts = np.array([-1.0, 1.0], dtype=np.complex128)
    elif mod in ("QPSK", "8PSK"):
        m = 4 if mod == "QPSK" else 8
        phases = 2 * np.pi * np.arange(m) / m
        pts = np.exp(1j * phases)
        pts = pts[np.argsort(_gray(m))]
    elif mod == "PAM4":
        pts = _pam_gray(4).astype(np.complex128)
    elif mod in ("16QAM", "64QAM"):
        m = 4 if mod == "16QAM" else 8
        lv = _pam_gray(m)
        pts = (lv[:, None] + 1j * lv[None, :]).reshape(-1)
    else:
        raise ValueError(f"unknown modulation {mod!r}")
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
    return pts.astype(np.complex128)


def bits_per_symbol(mod: str) -> int:
    return int(np.log2(len(constellation(mod))))


def symbols_to_bits(idx: np.ndarray, mod: str) -> np.ndarray:
    """Symbol indices -> bit matrix (n_sym, k), MSB first."""
    k = bits_per_symbol(mod)
    return ((idx[..., None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8)


# --------------------------------------------------------------------------
# Pulse shaping
# --------------------------------------------------------------------------


def rrc_taps(beta: float, sps: int, span: int) -> np.ndarray:
    """Root-raised-cosine filter, unit energy. Length = span*sps + 1."""
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    t = n / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4 * beta)):
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(
                np.pi * ti * (1 + beta)
            )
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h ** 2))


def upsample(sym: np.ndarray, sps: int) -> np.ndarray:
    out = np.zeros(len(sym) * sps, dtype=np.complex128)
    out[::sps] = sym
    return out


# --------------------------------------------------------------------------
# Channel and hardware impairments
# --------------------------------------------------------------------------


def tdl_channel(rng, n_taps: int = 3, decay_db: float = 6.0) -> np.ndarray:
    """Rayleigh tapped-delay-line with exponential power decay, unit energy."""
    p = 10 ** (-decay_db * np.arange(n_taps) / 10.0)
    p = p / p.sum()
    h = (rng.normal(size=n_taps) + 1j * rng.normal(size=n_taps)) / np.sqrt(2)
    h = h * np.sqrt(p)
    return h.astype(np.complex128)


def apply_cfo(x: np.ndarray, cfo_norm: float) -> np.ndarray:
    """Carrier frequency offset, cfo_norm in cycles per sample."""
    n = np.arange(len(x))
    return x * np.exp(2j * np.pi * cfo_norm * n)


def apply_phase_noise(x: np.ndarray, rng, linewidth: float) -> np.ndarray:
    """Wiener phase noise; linewidth is the per-sample phase std in radians."""
    if linewidth <= 0:
        return x
    phi = np.cumsum(rng.normal(scale=linewidth, size=len(x)))
    return x * np.exp(1j * phi)


def apply_saleh(x: np.ndarray, backoff_db: float) -> np.ndarray:
    """Saleh PA model driven at the given input backoff from saturation."""
    if backoff_db is None or not np.isfinite(backoff_db):
        return x
    scale = 10 ** (-backoff_db / 20.0) / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)
    r = np.abs(x) * scale
    a_a, b_a, a_p, b_p = 2.1587, 1.1517, 4.0033, 9.1040
    amp = a_a * r / (1 + b_a * r ** 2)
    pha = a_p * r ** 2 / (1 + b_p * r ** 2)
    return amp * np.exp(1j * (np.angle(x) + pha))


# --------------------------------------------------------------------------
# Sample synthesis
# --------------------------------------------------------------------------


def add_awgn(rng, x: np.ndarray, snr_db: float):
    """Add complex AWGN at the requested SNR relative to x's own power."""
    sig_p = np.mean(np.abs(x) ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    n = np.sqrt(noise_p / 2) * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
    return x + n, noise_p


def make_sample(
    rng,
    mod: str,
    snr_db: float,
    n_sym: int = 128,
    sps: int = 8,
    beta: float = 0.35,
    span: int = 8,
    multipath: bool = False,
    cfo_max: float = 0.0,
    phase_linewidth: float = 0.0,
    pa_backoff_db: float | None = None,
    phase_offset: bool = True,
    clean_mode: str = "rx",
):
    """Generate one paired (clean, noisy) example plus its metadata."""
    const = constellation(mod)
    idx = rng.integers(0, len(const), size=n_sym)
    sym = const[idx]

    g = rrc_taps(beta, sps, span)
    tx = np.convolve(upsample(sym, sps), g, mode="full")
    delay = (len(g) - 1) // 2
    tx = tx[delay : delay + n_sym * sps]           # align: symbol k at sample k*sps

    y = tx.copy()
    h = np.array([1.0 + 0j])
    if multipath:
        h = tdl_channel(rng)
        y = np.convolve(y, h, mode="full")[: len(tx)]
    cfo = float(rng.uniform(-cfo_max, cfo_max)) if cfo_max > 0 else 0.0
    if cfo != 0.0:
        y = apply_cfo(y, cfo)
    y = apply_phase_noise(y, rng, phase_linewidth)
    if pa_backoff_db is not None:
        y = apply_saleh(y, pa_backoff_db)
    # Unknown carrier phase. Without it a real-valued network can key on the
    # absolute I/Q axes, which is not information a receiver actually has, and
    # any complex-vs-real comparison is confounded.
    if phase_offset:
        y = y * np.exp(1j * rng.uniform(0, 2 * np.pi))

    p = np.sqrt(np.mean(np.abs(y) ** 2))
    y = y / p
    tx = tx / p

    clean = y if clean_mode == "rx" else tx
    noisy, noise_p = add_awgn(rng, y, snr_db)

    meta = dict(
        mod=mod, snr_db=float(snr_db), cfo=cfo, sps=sps, beta=beta, span=span,
        noise_var=float(noise_p),
    )
    return clean, noisy, sym, idx, h, meta


def build_dataset(
    out_path: str,
    n_per_cell: int = 64,
    mods=MODULATIONS,
    snrs=range(-20, 21, 2),
    seed: int = 0,
    n_sym: int = 128,
    sps: int = 8,
    n_source_blocks: int = 512,
    **kw,
):
    """
    Write an .npz dataset.

    `group` identifies the underlying source symbol block. The same block is
    reused across SNRs so that GroupKFold can guarantee a block never appears in
    both train and test - the single most common leakage path on RadioML-style
    data.
    """
    rng = np.random.default_rng(seed)
    clean, noisy, syms, idxs, chans, mod_id, snr_v, group, cfo_v = (
        [], [], [], [], [], [], [], [], []
    )
    mods = list(mods)
    snrs = list(snrs)
    for m_i, mod in enumerate(mods):
        for snr in snrs:
            for j in range(n_per_cell):
                gid = int(rng.integers(0, n_source_blocks))
                # offset the SNR: SeedSequence rejects negative entropy values
                sub = np.random.default_rng([seed, m_i, int(snr) + 128, j, gid])
                c, x, s, ix, h, meta = make_sample(
                    sub, mod, snr, n_sym=n_sym, sps=sps, **kw
                )
                clean.append(c.astype(np.complex64))
                noisy.append(x.astype(np.complex64))
                syms.append(s.astype(np.complex64))
                idxs.append(ix.astype(np.int16))
                chans.append(np.pad(h, (0, 8 - len(h))).astype(np.complex64))
                mod_id.append(m_i)
                snr_v.append(snr)
                group.append(gid)
                cfo_v.append(meta["cfo"])
    np.savez_compressed(
        out_path,
        clean=np.stack(clean), noisy=np.stack(noisy), symbols=np.stack(syms),
        sym_idx=np.stack(idxs), channel=np.stack(chans),
        mod_id=np.array(mod_id, np.int16), snr_db=np.array(snr_v, np.float32),
        group=np.array(group, np.int32), cfo=np.array(cfo_v, np.float32),
        mods=np.array(mods), sps=sps, n_sym=n_sym,
    )
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train.npz")
    ap.add_argument("--n-per-cell", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--multipath", action="store_true")
    ap.add_argument("--cfo-max", type=float, default=0.0)
    ap.add_argument("--phase-linewidth", type=float, default=0.0)
    ap.add_argument("--pa-backoff-db", type=float, default=None)
    ap.add_argument("--sps", type=int, default=8,
                    help="samples per symbol; sps=2 leaves the matched filter "
                         "only 3 dB of processing gain instead of 9 dB")
    ap.add_argument("--n-sym", type=int, default=128)
    ap.add_argument("--no-phase-offset", action="store_true")
    a = ap.parse_args()
    import os

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    build_dataset(
        a.out, n_per_cell=a.n_per_cell, seed=a.seed, sps=a.sps, n_sym=a.n_sym,
        multipath=a.multipath, cfo_max=a.cfo_max,
        phase_linewidth=a.phase_linewidth, pa_backoff_db=a.pa_backoff_db,
        phase_offset=not a.no_phase_offset,
    )
    import numpy as _np
    print(f"wrote {a.out}: L={a.sps*a.n_sym} samples, sps={a.sps}, "
          f"matched-filter gain {10*_np.log10(a.sps):.1f} dB")
