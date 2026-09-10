"""
Training entry point.

    python -m cvssd.train --data data/train.npz --model cvssd --protocol group
    python -m cvssd.train --data data/train.npz --model rvssd --match-params 48

The split protocol is mandatory and explicit; there is no random-split default,
because a random split over a RadioML-style file leaks source blocks between
train and test and inflates every number downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from cvssd.dataset import CompositeLoss, IQDenoiseDataset
from cvssd.models import MODELS, count_params, matched_real_width
from cvssd.splits import (check_no_leakage, group_and_snr_split, group_kfold,
                     leave_one_snr_out)

from tqdm import tqdm

def build_split(f, protocol: str, fold: int, held_out_snr: float, n_splits: int,
                seed: int):
    groups, snrs = f["group"], f["snr_db"]
    if protocol == "group":
        tr, te = list(group_kfold(groups, n_splits, seed))[fold]
    elif protocol == "snr":
        tr, te, _ = list(leave_one_snr_out(snrs, [held_out_snr]))[0]
    elif protocol == "both":
        tr, te = group_and_snr_split(groups, snrs, held_out_snr, n_splits, fold,
                                     seed)
    else:
        raise ValueError(protocol)
    assert len(te) > 0, "empty test split"
    if protocol in ("group", "both"):
        assert check_no_leakage(tr, te, groups), "group leakage detected"
    return np.asarray(tr), np.asarray(te)


@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    tot, n = 0.0, 0
    for noisy, clean, snr, _ in tqdm(loader, desc="Validation step", leave=True, ncols=80):
        noisy, clean, snr = noisy.to(device), clean.to(device), snr.to(device)
        est, snr_hat = model(noisy)
        loss, _ = crit(est, clean, snr_hat, snr)
        tot += float(loss) * noisy.shape[0]
        n += noisy.shape[0]
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="cvssd", choices=list(MODELS))
    p.add_argument("--out", default="runs")
    p.add_argument("--protocol", default="group", choices=["group", "snr", "both"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--held-out-snr", type=float, default=0.0)
    p.add_argument("--d-model", type=int, default=48)
    p.add_argument("--d-state", type=int, default=8)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--selection", default="mag", choices=["mag", "riparts"])
    p.add_argument("--match-params", type=int, default=None,
                   help="for rvssd: widen to match a complex model of this width")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--w-phase", type=float, default=0.1)
    p.add_argument("--w-spec", type=float, default=0.1)
    p.add_argument("--w-snr", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--tag", default=None)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full = IQDenoiseDataset(a.data)
    tr_idx, te_idx = build_split(full.f, a.protocol, a.fold, a.held_out_snr,
                                 a.n_splits, a.seed)
    train_ds = IQDenoiseDataset(a.data, tr_idx)
    val_ds = IQDenoiseDataset(a.data, te_idx)

    kw = dict(d_model=a.d_model, d_state=a.d_state, n_stages=a.stages,
              blocks_per_stage=a.blocks)
    if a.model in ("cvssd", "cvssd_wl", "cvssd_wleq"):
        kw["selection"] = a.selection
    if a.model == "rvssd" and a.match_params:
        d = matched_real_width(a.match_params, d_state=a.d_state,
                               n_stages=a.stages, blocks_per_stage=a.blocks)
        kw["d_model"] = d
        print(f"[match] real width {d} to match complex width {a.match_params}")
    SSM_MODELS = ("cvssd", "rvssd", "cvssd_wl", "cvssd_wleq")
    model = MODELS[a.model](**(kw if a.model in SSM_MODELS else {}))
    model = model.to(device)
    n_par = count_params(model)
    print(f"{a.model}: {n_par/1e6:.3f}M params | train {len(train_ds)} "
          f"| val {len(val_ds)} | device {device}")

    dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=a.batch_size, num_workers=a.workers)

    crit = CompositeLoss(a.w_phase, a.w_spec, a.w_snr)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=max(a.epochs * len(dl), 1), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and device == "cuda")

    tag = a.tag or f"{a.model}_{a.protocol}{a.fold}"
    run = os.path.join(a.out, tag)

    try:
        os.makedirs(run, exist_ok=False)
    except Exception as e:
        print(e)
        return 

    json.dump({**vars(a), "n_params": n_par},
              open(os.path.join(run, "config.json"), "w"), indent=2)

    best = float("inf")
    for ep in tqdm(range(a.epochs), desc="Training", leave=True, ncols=100):
        model.train()
        t0, run_loss = time.time(), 0.0
        for noisy, clean, snr, _ in tqdm(dl, desc="Train step", leave=True, ncols=80):
            noisy, clean, snr = (noisy.to(device, non_blocking=True),
                                 clean.to(device, non_blocking=True),
                                 snr.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            # Complex ops are kept in fp32: half-precision complex support is
            # incomplete and the scan is sensitive to underflow in exp(dt*A).
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                est, snr_hat = model(noisy)
                loss, parts = crit(est.float(), clean, snr_hat.float(), snr)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss)
        vloss = validate(model, vl, crit, device)
        print(f"ep {ep+1:3d}/{a.epochs}  train {run_loss/len(dl):.5f}  "
              f"val {vloss:.5f}  {time.time()-t0:.1f}s")
        if vloss < best:
            best = vloss
            torch.save({"model": model.state_dict(), "args": vars(a),
                        "kw": kw, "test_idx": te_idx},
                       os.path.join(run, "best.pt"))
    print("best val", best, "->", run)


if __name__ == "__main__":
    main()
