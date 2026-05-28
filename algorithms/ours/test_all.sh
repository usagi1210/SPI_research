#!/bin/bash
# Run test.py on Set11 + BSD68 for all sampling rates.
# Automatically finds the best checkpoint under results/ours/<model>/.
# Usage:
#   bash test_all.sh                        # tests Base-DUN
#   bash test_all.sh shared_dun             # tests Shared-DUN
#   bash test_all.sh base_dun configs/base_dun.yaml 7

MODEL=${1:-base_dun}
CONFIG=${2:-configs/${MODEL}.yaml}
GPU=${3:-7}
RESULT_BASE=../../results/ours/${MODEL}

echo "========================================"
echo "  Model : ${MODEL}"
echo "  Config: ${CONFIG}"
echo "  GPU   : ${GPU}"
echo "========================================"

for CR in 0.01 0.04 0.10 0.25 0.40 0.50; do
    CR_PCT=$(python3 -c "print(int(round(${CR}*100)))")
    CR_DIR="${RESULT_BASE}/cr${CR_PCT}"

    CKPT=$(find "${CR_DIR}" -name "best_cr${CR_PCT}.pth" -printf '%T@ %p\n' \
           2>/dev/null | sort -n | tail -1 | awk '{print $2}')

    if [ -z "$CKPT" ]; then
        echo "  [SKIP] No best checkpoint found for CR=${CR} (${CR_DIR})"
        continue
    fi

    echo ""
    echo "--- CR=${CR}  ckpt: ${CKPT} ---"
    python test.py --config "$CONFIG" --cr "$CR" --ckpt "$CKPT" \
        --gpu "$GPU" --bsd68
done

echo ""
echo "All done."
