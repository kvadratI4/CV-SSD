"""
Evaluation metrics for IQ denoising.

Two families:

1. Waveform-domain    : NMSE (dB), output SNR, SNR gain.
2. Communication-domain: EVM (%) and BER after a fixed, genie-aided receiver.

The receiver is deliberately identical for the noisy input, every denoiser under
comparison, and the clean reference. It performs only CFO removal, ZF
equalisation with the known channel, matched filtering, symbol-rate sampling and
pilot-based gain/phase correction. Nothing adaptive is learned inside it, so any
difference in EVM or BER is attributable to the denoiser alone.

Reporting EVM and BER - not just MSE - is what separates this evaluation from
most Mamba-denoising papers, which report reconstruction error only.
"""

from __future__ import annotations

import numpy as np

from cvssd.datagen import constellation, rrc_taps, symbols_to_bits, bits_per_symbol


# --------------------------------------------------------------------------
# Waveform-domain metrics
# --------------------------------------------------------------------------


def nmse_db(clean: np.ndarray, est: np.ndarray) -> float:
    """Normalised MSE in dB. Lower is better."""
    num = np.sum(np.abs(clean - est) ** 2)
    den = np.sum(np.abs(clean) ** 2) + 1e-20
    return float(10 * np.log10(num / den + 1e-20))


def output_snr_db(clean: np.ndarray, est: np.ndarray) -> float:
    """Output SNR in dB, i.e. -NMSE."""
    return -nmse_db(clean, est)


def snr_gain_db(clean: np.ndarray, noisy: np.ndarray, est: np.ndarray) -> float:
    """Improvement in output SNR over the unprocessed input, in dB."""
    return output_snr_db(clean, est) - output_snr_db(clean, noisy)


# --------------------------------------------------------------------------
# Receiver
# --------------------------------------------------------------------------


def equalize(x: np.ndarray, h: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Regularised ZF equalisation with a known channel response."""
    h = np.trim_zeros(np.asarray(h), "b")
    if h.size <= 1:
        return x / (h[0] if h.size else 1.0)
    n = len(x)
    H = np.fft.fft(h, n)
    return np.fft.ifft(np.fft.fft(x) * np.conj(H) / (np.abs(H) ** 2 + eps))


def recover_symbols(
    x: np.ndarray,
    *,
    mod: str,
    sps: int,
    n_sym: int,
    beta: float = 0.35,
    span: int = 8,
    cfo: float = 0.0,
    channel: np.ndarray | None = None,
    ref_symbols: np.ndarray | None = None,
    n_pilot: int = 16,
) -> np.ndarray:
    """Waveform -> soft symbol estimates, scaled to the unit-power constellation."""
    y = x
    if cfo:
        y = y * np.exp(-2j * np.pi * cfo * np.arange(len(y)))
    if channel is not None:
        y = equalize(y, channel)

    g = rrc_taps(beta, sps, span)
    mf = np.convolve(y, g, mode="full")
    d = (len(g) - 1) // 2
    k = np.arange(n_sym) * sps + d
    k = k[k < len(mf)]
    s = mf[k]

    if ref_symbols is not None and n_pilot > 0:
        p = min(n_pilot, len(s), len(ref_symbols))
        num = np.vdot(s[:p], ref_symbols[:p])       # conj(s) . ref
        den = np.vdot(s[:p], s[:p]).real + 1e-20
        s = s * (num / den)
    else:
        s = s / (np.sqrt(np.mean(np.abs(s) ** 2)) + 1e-20)
    return s


def hard_decide(s: np.ndarray, mod: str) -> np.ndarray:
    const = constellation(mod)
    return np.argmin(np.abs(s[:, None] - const[None, :]), axis=1)


def evm_percent(s: np.ndarray, ref: np.ndarray) -> float:
    """RMS EVM in percent against the reference symbols."""
    n = min(len(s), len(ref))
    err = np.mean(np.abs(s[:n] - ref[:n]) ** 2)
    p = np.mean(np.abs(ref[:n]) ** 2) + 1e-20
    return float(100 * np.sqrt(err / p))


def evm_db(s: np.ndarray, ref: np.ndarray) -> float:
    return float(20 * np.log10(evm_percent(s, ref) / 100 + 1e-20))


def ber(s: np.ndarray, ref_idx: np.ndarray, mod: str) -> tuple[int, int]:
    """Return (bit errors, total bits)."""
    n = min(len(s), len(ref_idx))
    dec = hard_decide(s[:n], mod)
    b_hat = symbols_to_bits(dec, mod)
    b_ref = symbols_to_bits(ref_idx[:n].astype(np.int64), mod)
    return int(np.sum(b_hat != b_ref)), int(b_ref.size)


def evaluate_sample(
    clean, noisy, est, *, mod, sps, n_sym, symbols, sym_idx,
    cfo=0.0, channel=None, beta=0.35, span=8, guard=8, n_pilot=8,
):
    """
    Full metric set for one example. `est` is the denoiser output.

    The first and last `guard` symbols are excluded from EVM and BER: their
    pulse tails are truncated by the block boundary, and the pilot symbols used
    for gain/phase correction live inside the leading guard, so measuring on
    them would be self-referential.
    """
    kw = dict(mod=mod, sps=sps, n_sym=n_sym, beta=beta, span=span,
              cfo=cfo, channel=channel, ref_symbols=symbols, n_pilot=n_pilot)
    sl = slice(guard, n_sym - guard) if guard else slice(None)
    s_est = recover_symbols(est, **kw)[sl]
    s_noisy = recover_symbols(noisy, **kw)[sl]
    symbols, sym_idx = symbols[sl], sym_idx[sl]
    e_est, t = ber(s_est, sym_idx, mod)
    e_noisy, _ = ber(s_noisy, sym_idx, mod)
    return dict(
        nmse_db=nmse_db(clean, est),
        out_snr_db=output_snr_db(clean, est),
        snr_gain_db=snr_gain_db(clean, noisy, est),
        evm_pct=evm_percent(s_est, symbols),
        evm_pct_noisy=evm_percent(s_noisy, symbols),
        bit_errors=e_est, bit_errors_noisy=e_noisy, n_bits=t,
        bits_per_symbol=bits_per_symbol(mod),
    )


def aggregate(rows, keys=("mod", "snr_db")):
    """Aggregate per-sample dicts into per-(mod, SNR) means with pooled BER."""
    from collections import defaultdict

    buck = defaultdict(list)
    for r in rows:
        buck[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in sorted(buck.items()):
        e = sum(r["bit_errors"] for r in rs)
        e0 = sum(r["bit_errors_noisy"] for r in rs)
        n = sum(r["n_bits"] for r in rs)
        rec = dict(zip(keys, key))
        rec.update(
            n=len(rs),
            nmse_db=float(np.mean([r["nmse_db"] for r in rs])),
            snr_gain_db=float(np.mean([r["snr_gain_db"] for r in rs])),
            evm_pct=float(np.mean([r["evm_pct"] for r in rs])),
            evm_pct_noisy=float(np.mean([r["evm_pct_noisy"] for r in rs])),
            ber=e / max(n, 1), ber_noisy=e0 / max(n, 1),
        )
        out.append(rec)
    return out
