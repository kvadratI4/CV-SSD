# The decisive experiment: complex CV-SSD vs its parameter-matched real twin.
# Identical data, splits, schedule and seeds. Run before anything else.
$ErrorActionPreference = "Stop" 

$DATA = "data/train.npz"
$TEST = "data/test.npz"
$D = 48
$EPOCHS = 40
$SEEDS = @("0", "1", "2")

foreach ($s in $SEEDS) {
    python -m cvssd.train --data "$DATA" --model cvssd --d-model $D `
        --epochs $EPOCHS --seed $s --protocol group --amp --tag "cvssd_s$s"

    python -m cvssd.train --data "$DATA" --model rvssd --match-params $D `
        --epochs $EPOCHS --seed $s --protocol group --amp --tag "rvssd_s$s"
}

foreach ($s in $SEEDS) {
    foreach ($m in @("cvssd", "rvssd")) {
        Write-Host "=== $m seed $s ==="
        python -m cvssd.evaluate --ckpt "runs/${m}_s$s/best.pt" --data "$TEST"
    }
}