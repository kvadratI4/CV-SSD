# Baselines. Five comparators is the minimum a reviewer will accept.
$ErrorActionPreference = "Stop" 

$DATA   = "data/train.npz"
$TEST   = "data/test.npz"
$EPOCHS = 40

foreach ($m in @("dncnn", "unet", "dae", "bilstm")) {
    python -m cvssd.train --data "$DATA" --model $m --epochs $EPOCHS `
        --protocol group --amp --tag "${m}_s0"

    python -m cvssd.evaluate --ckpt "runs/${m}_s0/best.pt" --data "$TEST"
}