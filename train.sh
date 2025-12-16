#!/usr/bin/env bash
set -euo pipefail

uv run python3 train.py \
  --map-size  10 \
  --episodes 1000 \
  --batch-size 16 \
  --target-path 8 \
  --checkpoint-dir model_output/dqn \
  --save-every 50 \
  --render \
  --enable-wandb \
  --project dqn-debug