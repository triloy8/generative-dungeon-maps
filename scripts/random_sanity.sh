#!/usr/bin/env bash
set -euo pipefail

uv run python3 inference.py \
  --algo random \
  --map-size 20 \
  --episodes 3 \
  --target-path 20 \
  --save-dir inference_output_random \
  --prob-empty 0.2 \
  --change-percentage 0.5 \
  --device cuda \
  --render \
  --random-mode carve \
  --dtype float32 \
  --seed 42
