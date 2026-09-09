"""
Leakage-controlled splitting.

Random splits on RadioML-style data inflate results because the same underlying
source waveform recurs across SNR cells. Two protocols are provided and both
should be reported:

  * group_kfold        - a source block never straddles train and test.
  * leave_one_snr_out  - held-out SNR never seen in training (generalisation
                         across noise level, the harder and more honest test).

A combined variant enforces both constraints at once.
"""

from __future__ import annotations

import numpy as np


def group_kfold(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Yield (train_idx, test_idx) with disjoint group ids, balanced by size."""
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    uniq, counts = uniq[perm], counts[perm]

    load = np.zeros(n_splits, dtype=np.int64)
    assign = {}
    for g, c in zip(uniq, counts):          # greedy least-loaded fold
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += c
    fold_of = np.array([assign[g] for g in groups])
    for f in range(n_splits):
        test = np.flatnonzero(fold_of == f)
        train = np.flatnonzero(fold_of != f)
        yield train, test


def leave_one_snr_out(snr_db: np.ndarray, held_out=None):
    """Yield (train_idx, test_idx, snr) holding out one SNR level at a time."""
    snr_db = np.asarray(snr_db)
    levels = np.unique(snr_db) if held_out is None else np.asarray(held_out)
    for s in levels:
        test = np.flatnonzero(snr_db == s)
        train = np.flatnonzero(snr_db != s)
        yield train, test, float(s)


def group_and_snr_split(groups, snr_db, held_out_snr, n_splits=5, fold=0, seed=0):
    """Hold out one SNR level *and* one group fold simultaneously."""
    tr_g, te_g = list(group_kfold(groups, n_splits, seed))[fold]
    snr_db = np.asarray(snr_db)
    te = np.array([i for i in te_g if snr_db[i] == held_out_snr], dtype=np.int64)
    tr = np.array([i for i in tr_g if snr_db[i] != held_out_snr], dtype=np.int64)
    return tr, te


def check_no_leakage(train_idx, test_idx, groups) -> bool:
    """Assert that no group id is shared between the two index sets."""
    g = np.asarray(groups)
    return len(np.intersect1d(g[train_idx], g[test_idx])) == 0
