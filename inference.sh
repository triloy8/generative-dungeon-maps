#!/usr/bin/env bash
set -euo pipefail

uv run python3 inference.py \
  --checkpoint model_output/dqn/weights_1000.safetensors \
  --map-size 14 \
  --episodes 3 \
  --target-path 7 \
  --render \
  --save-dir inference_output \
  --memory-capacity 10000 \
  --gamma 0.95 \
  --epsilon-start 1.0 \
  --epsilon-decay 0.999 \
  --epsilon-min 0.01 \
  --learning-rate 0.00005 \
  --clip-min -10.0 \
  --clip-max 10.0 \
  --target-update-interval 2000
