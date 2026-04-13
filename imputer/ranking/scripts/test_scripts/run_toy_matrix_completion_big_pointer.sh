#!/usr/bin/env bash
# Big model + --mc-entry-grid-pointer-rels (R=6).
# Optional: STEPS=50000 ./run_toy_matrix_completion_big_pointer.sh
# Output: under OUTPUT/toy_matrix_completion_ablation_fresh/ (override with OUT_DIR or TOY_MC_OUT_ROOT).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

STEPS="${STEPS:-10000}"
OUT_ROOT="${TOY_MC_OUT_ROOT:-OUTPUT/toy_matrix_completion_ablation_fresh}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/big_with_pointer}"

python toy_scripts/toy_matrix_completion.py \
  --steps "${STEPS}" \
  --num-train-graphs 32 \
  --num-test-graphs 5 \
  --N 4 --M 4 --D 1 \
  --mask-rate 0.3 \
  --num-layers 8 --attn-heads 8 --embedding-dim 128 --d-ff 384 \
  --resample-train-each-step \
  --mc-entry-grid-pointer-rels \
  --out-dir "${OUT_DIR}"
