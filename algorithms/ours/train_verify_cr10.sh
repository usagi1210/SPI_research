#!/bin/bash
# Quick verification: train all three stages at CR=10% only.
# Run this after the residual learning change to confirm all models
# train correctly before launching full 6-CR training.
#
# Usage: bash train_verify_cr10.sh [GPU]
# Default GPU: 4

GPU=${1:-4}
CR=0.10

echo "========================================"
echo "  Verification training at CR=${CR}"
echo "  GPU=${GPU}"
echo "========================================"

echo ""
echo "--- Stage 0: Base-DUN ---"
python train.py --config configs/base_dun.yaml    --cr $CR --gpu $GPU

echo ""
echo "--- Stage 1: Shared-DUN ---"
python train.py --config configs/shared_dun.yaml  --cr $CR --gpu $GPU

echo ""
echo "--- Stage 2: LoRA-DUN-fixed ---"
python train.py --config configs/lora_dun_fixed.yaml --cr $CR --gpu $GPU

echo ""
echo "All done. Check PSNR curves before launching full training."
