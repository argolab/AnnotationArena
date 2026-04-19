#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHONPATH=. python toy_scripts/toy_matrix_completion.py \
  --steps 50000 \
  --N 4 \
  --M 4 \
  --D 1 \
  --embedding-dim 512 \
  --num-layers 6 \
  --d-ff 2048 \
  --num-train-graphs 64 \
  --eval-every 50 \
  --out-dir OUTPUT/grok_ablation_4x4_r1_s50000_small/entity_marformer_h256_l6_ff1024_nt32_eval50_batched_r30k
