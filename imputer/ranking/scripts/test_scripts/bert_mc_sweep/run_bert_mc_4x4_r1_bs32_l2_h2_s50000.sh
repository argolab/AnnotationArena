#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHONPATH=. python toy_scripts/bert_toy_matrix_completion.py \
  --steps 50000 \
  --N 4 \
  --M 4 \
  --rank 1 \
  --train-batch-size 32 \
  --eval-batch-size 64 \
  --num-layers 2 \
  --num-heads 2 \
  --lr 1e-4 \
  --live-curves-every 100 \
  --out-dir OUTPUT/bert_mc_sweep/bert_4x4_r1_bs32_l2_h2_s50000

