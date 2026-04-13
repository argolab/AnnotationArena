#!/bin/bash
# Fast sanity run — MARFORMER-style defaults; toy enables use_rel_value (see deep wrapper comments).
# Small model/matrix and no --resample-train-each-step.

set -e

PYTHONPATH=. python toy_scripts/toy_matrix_completion.py \
  --steps 4000 \
  --num-train-graphs 48 \
  --num-test-graphs 8 \
  --N 5 \
  --M 5 \
  --D 2 \
  --mask-rate 0.3 \
  --embedding-dim 32 \
  --num-layers 2 \
  --attn-heads 4 \
  --d-ff 96 \
  --num-ffn-layers 1 \
  --dropout 0.1 \
  --type-embedding-init kaiming \
  --weight-decay 0.01 \
  --out-dir OUTPUT/toy_matrix_completion_curves_quick
