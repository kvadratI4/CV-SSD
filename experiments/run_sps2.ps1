# SPS2 ablations: CV-SSD vs RV-SSD, and WL / WLEQ variants.
# Runs everything in one go.
$ErrorActionPreference = "Stop"

# ---- Параметры ----
$TRAIN = "data/train_sps2.npz"
$TEST  = "data/test_sps2.npz"
$D     = 48
$EPOCHS = 40
$SEEDS = @("0")

# ============================================================
# Часть 1: cvssd vs rvssd
# ============================================================
Write-Host "`n########## Part 1: cvssd vs rvssd ##########" -ForegroundColor Cyan

foreach ($s in $SEEDS) {
    python -m cvssd.train --data $TRAIN --model cvssd --d-model $D `
        --epochs $EPOCHS --seed $s --protocol group --amp --tag "b_cvssd_s$s" --out "runs_pr2"

    python -m cvssd.train --data $TRAIN --model rvssd --match-params $D `
        --epochs $EPOCHS --seed $s --protocol group --amp --tag "b_rvssd_s$s" --out "runs_pr2"
}

$runs1 = @(
    "b_cvssd_s0", "b_cvssd_s1", "b_cvssd_s2",
    "b_rvssd_s0", "b_rvssd_s1", "b_rvssd_s2"
)
foreach ($r in $runs1) {
    Write-Host "=== evaluate $r ===" -ForegroundColor Yellow
    python -m cvssd.evaluate --ckpt "runs_pr2/$r/best.pt" --data $TEST
}

# ============================================================
# Часть 2: cvssd_wleq vs cvssd_wl
# ============================================================
Write-Host "`n########## Part 2: cvssd_wleq vs cvssd_wl ##########" -ForegroundColor Cyan

foreach ($s in $SEEDS) {
    python -m cvssd.train --data $TRAIN --model cvssd_wleq --d-model $D `
        --epochs $EPOCHS --seed $s --protocol group --amp --tag "b_wleq_s$s" --out "runs_pr3"

    python -m cvssd.train --data $TRAIN --model cvssd_wl --d-model $D `
        --epochs $EPOCHS --seed $s --protocol group --amp --tag "b_wl_s$s" --out "runs_pr3"
}

$runs2 = @(
    "b_wleq_s0", "b_wleq_s1", "b_wleq_s2",
    "b_wl_s0",   "b_wl_s1",   "b_wl_s2"
)
foreach ($r in $runs2) {
    Write-Host "=== evaluate $r ===" -ForegroundColor Yellow
    python -m cvssd.evaluate --ckpt "runs_pr3/$r/best.pt" --data $TEST
}

Write-Host "`nAll done." -ForegroundColor Green