#!/usr/bin/env bash
set -euo pipefail

uv run python3 inference.py \
  --checkpoint model_output/dqn/weights_1000.safetensors \
  --map-size 10 \
  --episodes 3 \
  --target-path 5 \
  --render \
  --save-dir inference_output \
  --prob-empty 0.5 \
  --change-percentage 0.2 \
  --device cuda \
  --dtype float32
