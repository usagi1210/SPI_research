#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_full.py \
  --config configs/full_shared_dun.yaml \
  --distributed
