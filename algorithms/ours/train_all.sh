#!/bin/bash
# Train Base-DUN for all sampling rates sequentially.
# Usage: bash train_all.sh
# Log files: ../../results/ours/base_dun/cr{N}/<run_id>/logs/

GPU=7
CONFIG=configs/base_dun.yaml

for CR in 0.01 0.04 0.10 0.25 0.40 0.50; do
    echo "========================================"
    echo "  CR=${CR}  GPU=${GPU}"
    echo "========================================"
    python train.py --config $CONFIG --cr $CR --gpu $GPU
done

echo "All sampling rates done."
