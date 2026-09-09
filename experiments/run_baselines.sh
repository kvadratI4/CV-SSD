#!/usr/bin/env bash
# Baselines. Five comparators is the minimum a reviewer will accept.
set -euo pipefail
DATA=${DATA:-data/train.npz}; TEST=${TEST:-data/test.npz}; EPOCHS=${EPOCHS:-40}
for m in dncnn unet dae bilstm; do
  python -m cvssd.train --data "$DATA" --model $m --epochs $EPOCHS \
      --protocol group --amp --tag "${m}_s0"
  python -m cvssd.evaluate --ckpt "runs/${m}_s0/best.pt" --data "$TEST"
done
