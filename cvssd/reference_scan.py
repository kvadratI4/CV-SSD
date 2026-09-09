"""
NumPy reference for the complex-valued selective scan.

The recurrence at the heart of a selective SSM is first-order and linear:

    h_t = a_t * h_{t-1} + b_t ,      a_t, b_t, h_t in C

For S4/S4D and Mamba-3 the state transition is complex diagonal, so this is
exactly the scalar complex recurrence above, replicated over (channel, state)
pairs. The associative operator

    (a1, b1) . (a2, b2) = (a1*a2,  a2*b1 + b2)

is associative over C, which is what licenses the log-depth Hillis-Steele scan
used in the PyTorch implementation. This module exists so that the parallel
GPU version can be checked against an unambiguous sequential ground truth.
"""

from __future__ import annotations

import numpy as np


def sequential_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Naive O(L) reference. a, b: (..., L). Returns h with the same shape."""
    h = np.zeros_like(b)
    acc = np.zeros(b.shape[:-1], dtype=b.dtype)
    for t in range(b.shape[-1]):
        acc = a[..., t] * acc + b[..., t]
        h[..., t] = acc
    return h


def parallel_scan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Log-depth Hillis-Steele inclusive scan of the same recurrence."""
    a, b = a.copy(), b.copy()
    L = b.shape[-1]
    step = 1
    while step < L:
        a_sh = np.concatenate(
            [np.ones_like(a[..., :step]), a[..., : L - step]], axis=-1
        )
        b_sh = np.concatenate(
            [np.zeros_like(b[..., :step]), b[..., : L - step]], axis=-1
        )
        b = b + a * b_sh
        a = a * a_sh
        step *= 2
    return b


def discretize(delta: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Zero-order-hold discretisation, Abar = exp(delta * A), A complex."""
    return np.exp(delta * A)


def stable_A(log_re: np.ndarray, im: np.ndarray) -> np.ndarray:
    """
    Complex diagonal state matrix with guaranteed |exp(delta*A)| < 1.

    Re(A) = -exp(log_re) < 0 forces contraction; Im(A) sets the oscillation
    frequency of the mode, which is what lets a single complex state track a
    rotating IQ phasor - the property a real-valued state cannot represent
    without spending two coupled channels on it.
    """
    return -np.exp(log_re) + 1j * im
