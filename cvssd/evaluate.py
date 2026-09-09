"""
Evaluation: per-(modulation, SNR) NMSE, SNR gain, EVM and BER.

    python -m cvssd.evaluate --ckpt runs/cvssd_group0/best.pt --data data/test.npz

Writes a tidy CSV. Every model is scored through the identical genie-aided
receiver in cvssd.metrics, so BER differences are attributable to denoising and
nothing else.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from cvssd.dataset import IQDenoiseDataset
from cvssd.metrics import aggregate, evaluate_sample
from cvssd.models import MODELS, count_params, matched_real_width


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a, kw = ck["args"], ck.get("kw", {})
    name = a["model"]
    if name == "rvssd" and a.get("match_params"):
        kw["d_model"] = matched_real_width(
            a["match_params"], d_state=a["d_state"], n_stages=a["stages"],
            blocks_per_stage=a["blocks"])
    model = MODELS[name](**(kw if name in ("cvssd", "rvssd") else {}))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck, name


@torch.no_grad()
def run(model, ds, device, batch_size=64):
    rows = []
    for start in range(0, len(ds), batch_size):
        sl = range(start, min(start + batch_size, len(ds)))
        noisy = torch.stack([ds[i][0] for i in sl]).to(device)
        est, _ = model(noisy)
        est = est.cpu().numpy()
        for k, i in enumerate(sl):
            m = ds.meta(i)
            j = ds.idx[i]
            clean = ds.f["clean"][j]
            nz = ds.f["noisy"][j]
            e = est[k, 0] + 1j * est[k, 1]
            r = evaluate_sample(
                clean, nz, e, mod=m["mod"], sps=m["sps"], n_sym=m["n_sym"],
                symbols=m["symbols"], sym_idx=m["sym_idx"], cfo=m["cfo"],
                channel=m["channel"])
            r.update(mod=m["mod"], snr_db=m["snr_db"])
            rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--use-test-split", action="store_true",
                   help="restrict to the held-out indices stored in the ckpt")
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck, name = load_model(a.ckpt, device)
    idx = ck.get("test_idx") if a.use_test_split else None
    ds = IQDenoiseDataset(a.data, idx)
    print(f"{name}: {count_params(model)/1e6:.3f}M params | {len(ds)} samples")

    rows = run(model, ds, device, a.batch_size)
    agg = aggregate(rows)

    out = a.out or os.path.join(os.path.dirname(a.ckpt), "metrics.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)

    by_snr = aggregate(rows, keys=("snr_db",))
    print(f"\n{'SNR':>6} {'NMSE dB':>9} {'gain dB':>8} {'EVM %':>8} "
          f"{'EVM0 %':>8} {'BER':>10} {'BER0':>10}")
    for r in by_snr:
        print(f"{r['snr_db']:6.0f} {r['nmse_db']:9.2f} {r['snr_gain_db']:8.2f} "
              f"{r['evm_pct']:8.2f} {r['evm_pct_noisy']:8.2f} "
              f"{r['ber']:10.2e} {r['ber_noisy']:10.2e}")
    g = float(np.mean([r["snr_gain_db"] for r in by_snr]))
    print(f"\nmean SNR gain {g:.2f} dB   ->  {out}")


if __name__ == "__main__":
    main()
