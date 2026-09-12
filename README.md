# CV-SSD — Complex-Valued Selective State-Space Denoiser for IQ Radio Signals

Reference implementation for an experimental Scopus-level paper on deep learning
denoising of communication waveforms.

The contribution is a complex-valued selective state-space (Mamba/S4-family)
denoiser whose **primary task is waveform restoration**, with blind SNR
conditioning and an evaluation protocol driven by EVM and BER rather than
reconstruction error alone.

---

## Status of the code

| Component | State |
|---|---|
| `cvssd/datagen.py`, `metrics.py`, `splits.py`, `reference_scan.py` | validated — `tests/test_core_numpy.py` passes |
| `cvssd/complex_ops.py`, `cvssm.py`, `models.py`, `dataset.py`, `train.py`, `evaluate.py` | written against the validated NumPy reference, **not yet executed** |

**Run `python tests/test_torch.py` first.** It is written to fail loudly on the
things that break silently: scan correctness, chunk boundaries, parameter
matching, gradient flow, stability, phase equivariance, and overfitting a batch.
Fix anything it reports before spending GPU hours.

## Install

```bash
pip install -r requirements.txt
python tests/test_core_numpy.py          # numeric core (numpy only)
python tests/test_equivariance_numpy.py  # equivariance, primitive by primitive
python tests/test_torch.py               # models — run this before training
```

## Quickstart

```bash
# 1. paired (clean, noisy) data with realistic impairments
python -m cvssd.datagen --out data/train.npz --n-per-cell 64 \
    --multipath --cfo-max 1e-4 --phase-linewidth 1e-3 --pa-backoff-db 8
python -m cvssd.datagen --out data/test.npz  --n-per-cell 16 --seed 999 \
    --multipath --cfo-max 1e-4 --phase-linewidth 1e-3 --pa-backoff-db 8

# 2. train the proposed model
python -m cvssd.train --data data/train.npz --model cvssd --protocol group --amp

# 3. score it: per-SNR NMSE, SNR gain, EVM, BER
python -m cvssd.evaluate --ckpt runs/cvssd_group0/best.pt --data data/test.npz
```

## Findings so far (sps=8 arm, 3 seeds each)

- Complex and its parameter-matched real twin (4.26M vs 4.28M, 0.37% apart) are
  **indistinguishable in aggregate**: NMSE -11.220±0.074 vs -11.114±0.282 dB.
- The difference splits **exactly along constellation circularity**: complex wins
  on all four proper constellations (QPSK, 8PSK, 16QAM, 64QAM) and loses on both
  improper ones (BPSK +2.65 pp EVM, PAM4 +1.53 pp, both >2 sd). A real-valued
  network mixing I/Q is implicitly *widely linear*; strictly linear complex
  processing is provably suboptimal for improper signals (Picinbono & Chevalier,
  IEEE TSP 1995). Hence `wl="eq"` — see `RUN_PLAN.md` Priority 3.
- Both models are **worse than the matched filter alone** on BER (+2.3%, +2.5%),
  because at sps=8 the filter supplies 9 dB of processing gain for free. Hence
  the sps=2 arm.
- `snr_gain_db` rewards trivial shrinkage: feed the metric `est = a*noisy` and it
  reports ~20 dB of "gain" at -20 dB input. Use `excess_over_shrinkage_db` and
  `si_sdr_gain_db`, which both correctly report 0.00 for that estimator.

See `RUN_PLAN.md` for what to run next and in what order.

## The experiment that decides whether you have a paper

Everything else is secondary to this:

```bash
bash experiments/run_ablation.sh
```

It trains `cvssd` (complex, d=48) against `rvssd` (real twin, widened by
`matched_real_width` to the same parameter count) under identical data, splits,
schedule and seeds. Run it with at least 3 seeds.

- **Complex wins by a clear margin on EVM/BER** → proceed to the full study.
- **The two are within noise** → you do not have a paper as framed. Pivot to the
  blind-SNR-conditioning contribution or to the impairment-robustness study, both
  of which are still unclaimed in the literature.

Do not skip the parameter matching. A complex layer holds two real kernels, so an
unmatched comparison silently rewards capacity and a reviewer will say so.

## Design decisions worth defending in the paper

**Ground truth.** The clean target is the *noise-free received* waveform, not the
ideal transmit waveform. Using the latter would fold equalisation into the
denoising metric and make comparison against CNN/GAN baselines unfair. Switch
with `clean_mode="tx"` if you want that ablation.

**Phase equivariance.** Selection (Δ, B, C), the SNR head, FiLM and the block
gate all read magnitude only, and every complex layer is bias-free, so the model
satisfies `f(e^{iθ}x) = e^{iθ}f(x)` exactly. The gate in particular must stay
*real*: multiplying two complex equivariant branches yields `e^{2iθ}` and
silently destroys the property (see `tests/test_equivariance_numpy.py`, which
reproduces exactly that failure). Absolute carrier phase is nuisance information
in RF; the network is structurally forbidden from keying on it. `tests/test_torch.py`
verifies this numerically, the real twin provably lacks it, and
`--selection riparts` ablates it. This is a stronger theoretical claim than
"we combined two popular blocks".

**Why complex SSMs are the natural choice, not an arbitrary pairing.** S4 and S4D
already carry a *complex diagonal* state transition matrix, and Mamba-3
re-introduces complex state to recover state-tracking. A single complex state
tracks a rotating phasor that a real state can only represent by spending two
coupled channels. The signal is complex; the state was already complex; CV-SSD
removes an artificial split rather than adding machinery.

**Blind SNR.** The conditioning scalar comes from a learned magnitude-statistics
head (the same moments classical M2M4 estimators use), never from an oracle. Many
published denoisers quietly consume the true noise level; state explicitly that
yours does not.

**Evaluation.** `cvssd/metrics.py` runs one fixed, genie-aided receiver — CFO
removal, ZF equalisation with the known channel, matched filter, symbol-rate
sampling, pilot-based gain correction — identically for the noisy input, every
model and the clean reference. Nothing adaptive is learned inside it, so EVM and
BER differences are attributable to denoising alone. The first and last 8 symbols
are excluded (truncated pulse tails; the pilots live in the leading guard).

**Splits.** `--protocol group` (GroupKFold over source blocks), `--protocol snr`
(Leave-One-SNR-Out) and `--protocol both`. Report at least the first two. Random
splits on RadioML-style data leak source blocks and inflate every number.

## What to report

Per (modulation, SNR) and pooled:

- NMSE (dB), output SNR, **SNR gain (dB)** — headline number
- **EVM (%)** and **BER**, denoised vs unprocessed
- Parameters, FLOPs, latency on GPU and CPU — linear-time inference is the
  practical selling point of an SSM over a transformer denoiser
- Ablations: complex vs real twin (matched params) · `mag` vs `riparts` selection ·
  FiLM on/off · loss terms · `d_state` · chunk size

## Known limitations to state in the paper

- The RRC truncation leaves a systematic EVM floor near 4% for BPSK and 1.3% for
  64QAM, measured on the clean waveform. It is identical for every method, but
  report it so the high-SNR plateau is not mistaken for a model artefact.
- The receiver is genie-aided (known CFO and channel). Equalisation is out of
  scope; say so rather than letting a reviewer discover it.
- The scan is pure PyTorch. The official `mamba-ssm` CUDA kernel **cannot** be
  substituted: it does not support complex input. Expect to be slower than a
  fused real-valued Mamba; report wall-clock honestly and treat a complex fused
  kernel as future work.
- Only linear modulations (PSK/QAM/PAM) are generated, because EVM and BER are
  well defined for them. Adding GFSK/CPFSK requires a different symbol metric.

## Positioning against prior work

The nearest published work is **IQUMamba-1D** (Peng, *J. King Saud Univ. Comput.
Inf. Sci.* 2026, 38:63, DOI 10.1007/s44443-025-00440-5), which also combines
complex-valued selective state-space modelling with joint I/Q processing — but for
**single-channel blind source separation**, not denoising. Cite it prominently and
state the distinction explicitly; do not claim "first complex-valued Mamba on IQ".
The defensible claim is: first complex-valued selective SSM whose primary
objective is IQ waveform denoising, with phase-equivariant selection and blind SNR
conditioning, evaluated end-to-end on EVM and BER.

Real-valued Mamba in RF (MAMCA, ConvMamba, TCN-Mamba) uses denoising only as a
module subordinate to modulation classification. RF denoisers whose primary task
*is* denoising (DRdA-CA, RaGAN, CDDM) are all real-valued CNN/GAN/diffusion. That
combination — complex × selective SSM × RF × denoising-primary — is the empty cell
this repository targets.

## Layout

```
cvssd/
  datagen.py        paired data: PSK/QAM/PAM, RRC, TDL multipath, CFO,
                    phase noise, Saleh PA
  metrics.py        genie receiver, NMSE, SNR gain, EVM, BER
  splits.py         GroupKFold, Leave-One-SNR-Out, combined
  reference_scan.py NumPy ground truth for the complex recurrence
  complex_ops.py    complex conv/linear/norm/activations
  cvssm.py          complex & real selective SSM, chunked parallel scan, FiLM
  models.py         CV-SSD, real twin, DnCNN1D, UNet1D, ConvDAE, BiLSTM
  dataset.py        loader + composite loss (waveform, phase, spectral, SNR)
  train.py          leakage-controlled training
  evaluate.py       per-SNR metric tables
tests/              numpy core + equivariance suites (passing), torch suite
experiments/        ablation driver
```
