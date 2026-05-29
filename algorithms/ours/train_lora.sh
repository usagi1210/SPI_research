#!/bin/bash
# Train LoRA-DUN-fixed (Stage 2) — two-stage training.
# Backbone is loaded from the best SharedDUN checkpoint and frozen;
# only LoRA adapters and alpha step-sizes are optimised.
#
# Usage: bash train_lora.sh [GPU]
# Default GPU: 4

GPU=${1:-4}
CONFIG=configs/lora_dun_fixed.yaml
SHARED_RESULTS=../../results/ours/shared_dun

for CR in 0.01 0.04 0.10 0.25 0.40 0.50; do
    CR_PCT=$(python3 -c "print(int(round(${CR}*100)))")

    BACKBONE=$(find "${SHARED_RESULTS}/cr${CR_PCT}" -name "best_cr${CR_PCT}.pth" \
               -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | awk '{print $2}')

    if [ -z "$BACKBONE" ]; then
        echo "  [WARN] No SharedDUN checkpoint for CR=${CR}; training without backbone init"
        python train.py --config $CONFIG --cr $CR --gpu $GPU
    else
        echo "========================================"
        echo "  CR=${CR}  GPU=${GPU}"
        echo "  backbone: ${BACKBONE}"
        echo "========================================"
        python train.py --config $CONFIG --cr $CR --gpu $GPU \
            --backbone_ckpt "$BACKBONE" --freeze_backbone
    fi
done

echo "All sampling rates done."
