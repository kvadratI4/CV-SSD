#!/usr/bin/env bash
# The decisive experiment: complex CV-SSD vs its parameter-matched real twin.
# Identical data, splits, schedule and seeds. Run before anything else.
set -euo pipefail

DATA=${DATA:-data/train.npz}
TEST=${TEST:-data/test.npz}
D=${D:-48}
EPOCHS=${EPOCHS:-40}
SEEDS=${SEEDS:-"0 1 2"}

for s in $SEEDS; do
  python -m cvssd.train --data "$DATA" --model cvssd --d-model $D \
      --epochs $EPOCHS --seed $s --protocol group --amp --tag "cvssd_s$s"
  python -m cvssd.train --data "$DATA" --model rvssd --match-params $D \
      --epochs $EPOCHS --seed $s --protocol group --amp --tag "rvssd_s$s"
done

for s in $SEEDS; do
  for m in cvssd rvssd; do
    echo "=== $m seed $s ==="
    python -m cvssd.evaluate --ckpt "runs/${m}_s$s/best.pt" --data "$TEST"
  done
done
