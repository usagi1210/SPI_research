#!/bin/bash
# 顺序训练所有采样率（单卡）
# 用法：bash train_all.sh [GPU_ID]
# 示例：bash train_all.sh 7

GPU=${1:-0}

for ratio in 1 4 10 25 40 50; do
    echo "========================================"
    echo "Training CS ratio=${ratio}%  GPU=${GPU}"
    echo "========================================"
    python train.py --cs_ratio $ratio --gpu $GPU
done

echo "All ratios done."
