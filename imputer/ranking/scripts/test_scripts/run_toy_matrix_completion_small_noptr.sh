#!/usr/bin/env bash
# Small model (4L / 4H / emb 32 / d_ff 96), no mc_entry same-row/col pointer rels.
# Optional: STEPS=50000 ./run_toy_matrix_completion_small_noptr.sh
# Output: under OUTPUT/toy_matrix_completion_ablation_fresh/ (override with OUT_DIR or TOY_MC_OUT_ROOT).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

STEPS="${STEPS:-10000}"
OUT_ROOT="${TOY_MC_OUT_ROOT:-OUTPUT/toy_matrix_completion_ablation_fresh}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/small_no_pointer}"

python toy_scripts/toy_matrix_completion.py \
  --steps "${STEPS}" \
  --num-train-graphs 32 \
  --num-test-graphs 5 \
  --N 4 --M 4 --D 1 \
  --mask-rate 0.3 \
  --num-layers 4 --attn-heads 4 --embedding-dim 32 --d-ff 96 \
  --resample-train-each-step \
  --out-dir "${OUT_DIR}"
