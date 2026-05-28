#!/bin/bash
# Resume Base-DUN training for all sampling rates.
# Automatically finds the latest checkpoint under results/ours/base_dun/cr{N}/.
# Usage: bash resume_all.sh
# Assumes the same GPU and config used in train_all.sh.

GPU=7
CONFIG=configs/base_dun.yaml
RESULT_BASE=../../results/ours/base_dun

for CR in 0.01 0.04 0.10 0.25 0.40 0.50; do
    CR_PCT=$(python3 -c "print(int(round(${CR}*100)))")
    CR_DIR="${RESULT_BASE}/cr${CR_PCT}"

    # Find the most recently modified latest_cr{N}.pth under any run_id subdir
    CKPT=$(find "${CR_DIR}" -name "latest_cr${CR_PCT}.pth" -printf '%T@ %p\n' \
           2>/dev/null | sort -n | tail -1 | awk '{print $2}')

    if [ -z "$CKPT" ]; then
        echo "  [SKIP] No checkpoint found for CR=${CR} (${CR_DIR})"
        continue
    fi

    # Extract run_id from path: .../cr{N}/{run_id}/checkpoints/latest_cr{N}.pth
    RUN_ID=$(echo "$CKPT" | awk -F'/' '{for(i=1;i<=NF;i++) if($i~/^[0-9]{8}_[0-9]{6}$/) print $i}')

    echo "========================================"
    echo "  CR=${CR}  run_id=${RUN_ID}  GPU=${GPU}"
    echo "  ckpt: ${CKPT}"
    echo "========================================"

    python train.py --config $CONFIG --cr $CR --gpu $GPU \
        --resume "$CKPT" --run_id "$RUN_ID"
done

echo "All sampling rates resumed."
