#!/usr/bin/env bash
set -euo pipefail

uv run python3 inference.py \
  --algo ppo \
  --checkpoint model_output/10_ppo/weights_1000.safetensors \
  --map-size 10 \
  --episodes 3 \
  --target-path 7 \
  --render \
  --save-dir inference_output_ppo \
  --prob-empty 0.5 \
  --change-percentage 0.2 \
  --device cuda \
  --dtype float32
