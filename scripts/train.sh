#!/usr/bin/env bash
set -euo pipefail

uv run python3 train.py \
  --algo ppo \
  --map-size  10 \
  --episodes 1000 \
  --target-path 5 \
  --checkpoint-dir model_output/ppo \
  --save-every 50 \
  --render \
  --enable-wandb \
  --project ppo-debug \
  --gamma 0.95 \
  --learning-rate 0.00005 \
  --rollout-steps 512 \
  --ppo-epochs 4 \
  --ppo-minibatch-size 64 \
  --ppo-clip-eps 0.2 \
  --gae-lambda 0.95 \
  --entropy-coef 0.01 \
  --value-coef 0.5 \
  --max-grad-norm 0.5 \
  --prob-empty 0.5 \
  --change-percentage 0.2 \
  --device cuda \
  --dtype float32
