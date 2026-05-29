#!/bin/bash
# Train LoRA-DUN-fixed (Stage 2) for all sampling rates sequentially.
# Usage: bash train_lora.sh

GPU=4
CONFIG=configs/lora_dun_fixed.yaml

for CR in 0.01 0.04 0.10 0.25 0.40 0.50; do
    echo "========================================"
    echo "  CR=${CR}  GPU=${GPU}"
    echo "========================================"
    python train.py --config $CONFIG --cr $CR --gpu $GPU
done

echo "All sampling rates done."
