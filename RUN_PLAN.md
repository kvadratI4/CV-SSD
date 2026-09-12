# Run plan

Ordered by value per GPU-hour. Do them in order and stop wherever time runs out
— the paper is writable after Priority 1, and each later block strengthens it.

The training objective is deliberately unchanged from your first six runs, so
`cvssd_s{0,1,2}` and `rvssd_s{0,1,2}` stay valid and go straight into the paper
as the sps=8 arm.

First, once:

```bash
python tests/test_torch.py        # now also covers the widely-linear variants
```

---

## Priority 1 — baselines on the existing data (arm A, sps=8)

Without these the paper is rejected on comparator count, whatever else it shows.
Four models × 1 seed is the minimum; add seeds 1 and 2 if time allows.

```bash
DATA=data/train.npz TEST=data/test.npz EPOCHS=40 bash experiments/run_baselines.sh
```

Gives: DnCNN1D, U-Net1D, ConvDAE, BiLSTM. Together with your CV-SSD, the real
twin, the matched filter and scalar shrinkage that is eight comparators.

## Priority 2 — arm B: sps=2 with unknown carrier phase

Two changes, both necessary for the conclusions to hold up:

- `sps=2` leaves the matched filter 3 dB of processing gain instead of 9 dB, so
  the task is no longer solvable by a linear filter.
- a random global phase per example, which your first dataset lacked. Without it
  a real-valued network can key on the absolute I/Q axes — information no real
  receiver has — and the complex-vs-real comparison is confounded.

```bash
python -m cvssd.datagen --out data/train_sps2.npz --n-per-cell 64 --seed 0 \
    --sps 2 --n-sym 512 --multipath --cfo-max 1e-4 \
    --phase-linewidth 1e-3 --pa-backoff-db 8
python -m cvssd.datagen --out data/test_sps2.npz --n-per-cell 16 --seed 999 \
    --sps 2 --n-sym 512 --multipath --cfo-max 1e-4 \
    --phase-linewidth 1e-3 --pa-backoff-db 8

for s in 0 1 2; do
  python -m cvssd.train --data data/train_sps2.npz --model cvssd --d-model 48 \
      --epochs 40 --seed $s --protocol group --amp --tag "b_cvssd_s$s"
  python -m cvssd.train --data data/train_sps2.npz --model rvssd --match-params 48 \
      --epochs 40 --seed $s --protocol group --amp --tag "b_rvssd_s$s"
done
for r in b_cvssd_s0 b_cvssd_s1 b_cvssd_s2 b_rvssd_s0 b_rvssd_s1 b_rvssd_s2; do
  python -m cvssd.evaluate --ckpt runs/$r/best.pt --data data/test_sps2.npz
done
```

**The question this answers:** does the properness split (complex wins on
QPSK/8PSK/QAM, loses on BPSK/PAM4) reproduce when the matched filter is no
longer doing the work and the carrier phase is unknown? If yes, the finding is
robust and the paper has a real result. If it vanishes, report that too — it
would mean the split was specific to the filter-dominated regime, which is
itself worth knowing.

## Priority 3 — the mechanism test: widely-linear variants

This is what turns the properness finding from a correlation into a
demonstrated mechanism, and it is the strongest single experiment available.

```bash
for s in 0 1 2; do
  python -m cvssd.train --data data/train_sps2.npz --model cvssd_wleq --d-model 48 \
      --epochs 40 --seed $s --protocol group --amp --tag "b_wleq_s$s"
  python -m cvssd.train --data data/train_sps2.npz --model cvssd_wl --d-model 48 \
      --epochs 40 --seed $s --protocol group --amp --tag "b_wl_s$s"
done
for r in b_wleq_s0 b_wleq_s1 b_wleq_s2 b_wl_s0 b_wl_s1 b_wl_s2; do
  python -m cvssd.evaluate --ckpt runs/$r/best.pt --data data/test_sps2.npz
done
```

`cvssd_wleq` adds the conjugate branch paired with the pseudo-covariance, which
keeps exact phase equivariance and self-gates (the branch vanishes on circular
constellations, `|u| = 0.00`, and is fully active on BPSK/PAM4, `|u| = 1.00` —
verified in `tests/test_equivariance_numpy.py`). `cvssd_wl` is the plain
conjugate branch, which works but forfeits equivariance.

**Prediction to state in the paper before looking:** `cvssd_wleq` closes the
BPSK/PAM4 gap against the real twin while keeping the QAM/PSK advantage. If it
holds, you have an architectural contribution grounded in classical improper-
signal theory rather than in "complex numbers suit I/Q".

If only one variant fits in the time you have, run `cvssd_wleq`.

## Priority 4 — cheap ablations, one seed each

```bash
python -m cvssd.train --data data/train_sps2.npz --model cvssd --d-model 48 \
    --selection riparts --epochs 40 --seed 0 --protocol group --amp --tag b_riparts_s0
python -m cvssd.train --data data/train_sps2.npz --model cvssd --d-model 48 \
    --w-snr 0 --epochs 40 --seed 0 --protocol group --amp --tag b_nosnr_s0
python -m cvssd.train --data data/train_sps2.npz --model cvssd --d-model 48 \
    --protocol snr --held-out-snr 0 --epochs 40 --seed 0 --amp --tag b_loso_s0
```

Covers the equivariance ablation (Section 6.2), the blind-SNR-conditioning
ablation (6.3) and the Leave-One-SNR-Out protocol, which one table must use or a
reviewer will ask why only GroupKFold appears.

---

## What to send back

Per run, just `runs/<tag>/metrics.csv` and `runs/<tag>/config.json`, all at once,
prefixed with the tag. Not `best.pt`. The training stdout is useful if you kept
it — the loss curve shows whether 40 epochs was enough.

`metrics.csv` now carries three extra columns: `si_sdr_db`, `si_sdr_gain_db` and
`excess_over_shrinkage_db`. The last one is the honest headline figure;
`snr_gain_db` is kept only so the paper can show what it conceals.

## Total cost

Priority 1 is 4 runs, Priority 2 is 6, Priority 3 is 6, Priority 4 is 3 — 19
runs at your current 40-epoch setting. Arm B sequences are 1024 samples at
`sps=2`, the same length as before, so per-epoch time should be comparable.
