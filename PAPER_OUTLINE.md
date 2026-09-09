# Manuscript outline — CV-SSD

Target length ~8–10k words. Each section below names the experiment that fills
it, so nothing is written before the number backing it exists.

---

## Title

*A Phase-Equivariant Complex-Valued Selective State-Space Network for Denoising
IQ Communication Waveforms*

Avoid "first" in the title. Put the priority claim in the contributions list,
where it can be qualified.

## Abstract (200–250 words)

Problem: denoising at low SNR is the bottleneck for downstream demodulation and
classification; existing RF denoisers are real-valued CNN/GAN/diffusion, and the
state-space models that reached RF treat denoising as a module inside a
classifier. Method: complex-valued selective SSM with magnitude-driven selection,
blind SNR conditioning, additive skips. Result: quote SNR gain in dB, EVM
reduction, and BER reduction at a named SNR, plus parameter/latency numbers.
Do not report classification accuracy as the headline — that is the metric the
prior work already optimises.

## 1. Introduction

- Low-SNR operation in cognitive radio, spectrum sensing, non-cooperative receivers.
- Three gaps, stated plainly: (i) SSMs in RF are classifiers, not denoisers;
  (ii) the RF denoisers that exist are real-valued and discard the I/Q coupling;
  (iii) evaluation is dominated by classification accuracy, which a denoiser can
  improve while distorting the constellation.
- Contributions, numbered, four of them:
  1. A complex-valued selective SSM denoiser for IQ waveforms — to our knowledge
     the first whose primary objective is waveform restoration rather than
     separation or classification.
  2. Magnitude-driven selection making the network **exactly phase-equivariant**,
     with proof and numerical verification.
  3. Blind SNR conditioning — no oracle noise level at inference.
  4. An evaluation protocol on EVM and BER through a fixed genie receiver, under
     leakage-controlled splits.

## 2. Related work

Four subsections; use the competitor table from the literature review.

- **RF denoising, primary task**: DRdA-CA (10.3390/s23021023), RaGAN
  (10.3390/s23010475), CDDM (arXiv:2305.09161), RF-Diffusion (10.1145/3636534.3649348).
  All real-valued CNN/GAN/diffusion.
- **State-space models in RF**: MAMCA (arXiv:2405.11263), ConvMamba
  (10.3390/app15179805), TCN–Mamba (10.3390/electronics15010188). All real-valued,
  denoising subordinate to AMC.
- **Complex-valued networks**: Trabelsi et al. (arXiv:1705.09792); CVNN for AMC
  and device ID; Fuchs et al. radar interference mitigation (RadarConf 2021).
- **Nearest work — must be explicit.** IQUMamba-1D (10.1007/s44443-025-00440-5)
  combines complex-valued selective SSM with joint I/Q, but for single-channel
  blind source separation. State the denoising-vs-separation distinction in its
  own paragraph. A reviewer who finds it before you do will reject the paper.

End with a table: axes = {complex-valued, SSM backbone, denoising-primary,
EVM/BER reported}, showing the empty cell.

## 3. Method

- **3.1 Signal model.** y = h*x + impairments + n; define the denoising target as
  the noise-free received waveform and justify why not the ideal transmit waveform.
- **3.2 Background.** S4/S4D complex diagonal state matrix (arXiv:2111.00396,
  arXiv:2206.11893); Mamba selection (arXiv:2312.00752); Mamba-3 re-introducing
  complex state (arXiv:2603.15569). This is the paragraph that makes CV-SSD look
  inevitable rather than arbitrary.
- **3.3 Complex selective SSM.** Discretisation Ā = exp(ΔA), stability from
  Re(A) < 0, chunked associative scan and why the fused `mamba-ssm` kernel is
  unusable (no complex support).
- **3.4 Phase equivariance.** State and prove the proposition
  f(e^{iθ}z) = e^{iθ}f(z): complex-linear layers, magnitude-only selection,
  magnitude-only conditioning, no biases. One short proof, three lines.
- **3.5 Architecture.** U-shape, additive learnable skips, blind SNR head + FiLM.
- **3.6 Loss.** Waveform + magnitude-weighted phase + STFT magnitude + SNR aux.

## 4. Experimental setup

- Data: synthetic paired set (TorchSig/GNU Radio style) with CFO, phase noise,
  TDL multipath, Saleh PA; plus RML22 or RadioML 2018.01A for comparability.
- Splits: GroupKFold and Leave-One-SNR-Out. State that random splits leak.
- Baselines: DnCNN1D, U-Net1D, ConvDAE, BiLSTM/GRU, real-valued SSD twin. Five
  minimum.
- Metrics: NMSE, SNR gain, EVM, BER, params/FLOPs/latency.
- Training details, 3 seeds, mean ± std everywhere.

## 5. Results

- **Table 1** SNR gain and NMSE per SNR, all methods.
- **Fig. 1** BER vs input SNR, all methods + unprocessed + theory curve.
- **Fig. 2** EVM vs SNR, and constellation scatter at −4 dB before/after.
- **Table 2** Efficiency: params, FLOPs, GPU and CPU latency vs sequence length.
  Include a transformer denoiser to show the linear-complexity advantage.
- **Fig. 3** Scaling with sequence length (1024 → 8192) — the SSM argument.

## 6. Ablations

- **6.1 Complex vs real twin at matched parameters.** The decisive table. If this
  is within noise, the paper is reframed, not published as is.
- **6.2 Selection: magnitude (equivariant) vs real/imag parts.** Include the
  measured equivariance error to show the property is real, not rhetorical.
- **6.3 Blind vs oracle SNR conditioning vs none.**
- **6.4 Loss terms.** 6.5 d_state and chunk size. 6.6 Additive vs concatenated skips.

## 7. Limitations

Genie-aided receiver; no equalisation; linear modulations only; no fused complex
kernel so wall-clock lags real-valued Mamba; RRC truncation EVM floor (~4% BPSK,
1.3% 64QAM). Stating these yourself is worth more than having them found.

## 8. Conclusion

---

## Figure/table checklist before submission

- [ ] Every number is mean ± std over ≥3 seeds
- [ ] Per-SNR curves, never only pooled averages
- [ ] Parameter counts printed for every compared model
- [ ] Split protocol named in every table caption
- [ ] Equivariance error reported numerically
- [ ] Code and data-generation scripts released, with the commit hash cited

## Submission order

1. MDPI Electronics or Sensors — fast, proven appetite for RF + Mamba.
2. Digital Signal Processing (Elsevier, Q2) — more prestige, good scope fit.
3. IEEE TCCN (Q1) — only if results beat every baseline and Section 3.4 is solid.

Russian Technological Journal is **not** Scopus-indexed; do not use it if a
Scopus venue is the requirement.
