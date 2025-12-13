#!/usr/bin/env bash
set -euo pipefail

uv run python3 inference.py \
  --checkpoint model_output/dqn/weights_0950.pt \
  --map-size 7 \
  --episodes 3 \
  --target-path 1 \
  --render \
  --save-dir inference_output
