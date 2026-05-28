#!/bin/bash
# Train Shared-DUN (Stage 1) for all sampling rates sequentially.
# Usage: bash train_shared.sh

GPU=7
CONFIG=configs/shared_dun.yaml

for CR in 0.01 0.04 0.10 0.25 0.40 0.50; do
    echo "========================================"
    echo "  CR=${CR}  GPU=${GPU}"
    echo "========================================"
    python train.py --config $CONFIG --cr $CR --gpu $GPU
done

echo "All sampling rates done."
