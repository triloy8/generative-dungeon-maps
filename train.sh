#!/usr/bin/env bash
set -euo pipefail

uv run python3 train.py \
  --map-size  7 \
  --episodes 1000 \
  --batch-size 16 \
  --target-path 4 \
  --checkpoint-dir model_output/dqn \
  --save-every 50 \
  --render \
  --enable-wandb \
  --project dqn-debug \
  --memory-capacity 10000 \
  --gamma 0.95 \
  --epsilon-start 1.0 \
  --epsilon-decay 0.999 \
  --epsilon-min 0.01 \
  --learning-rate 0.00005 \
  --clip-min -10.0 \
  --clip-max 10.0 \
  --target-update-interval 2000 \
  --prob-empty 0.5 \
  --change-percentage 0.2
