#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHONPATH=. python toy_scripts/bert_toy_matrix_completion.py \
  --steps 50000 \
  --N 6 \
  --M 6 \
  --rank 1 \
  --train-batch-size 256 \
  --eval-batch-size 64 \
  --lr 1e-4 \
  --live-curves-every 100 \
  --out-dir OUTPUT/bert_mc_sweep/bert_6x6_r1_s50000

