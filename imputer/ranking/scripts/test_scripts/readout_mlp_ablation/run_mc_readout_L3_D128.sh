#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

PYTHONPATH=. python toy_scripts/toy_matrix_completion.py \
  --steps 10000 \
  --seed 42 \
  --N 4 --M 4 --D 1 \
  --readout-mlp-layer 3 \
  --readout-mlp-dim 128 \
  --out-dir OUTPUT/mc_readout_mlp/L3_D128

