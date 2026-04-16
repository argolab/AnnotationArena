#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS="${STEPS:-50000}"
LIVE_CURVES_EVERY="${LIVE_CURVES_EVERY:-100}"
BASE_OUT_DIR="${BASE_OUT_DIR:-OUTPUT/grok_ablation_4x4_r1_s50000_bertlike}"

COMMON_ARGS=(
  --steps "${STEPS}"
  --live-curves-every "${LIVE_CURVES_EVERY}"
)

echo "Running 1/4: flat + discrete"
PYTHONPATH=. python toy_scripts/bert_toy_matrix_completion.py \
  "${COMMON_ARGS[@]}" \
  --out-dir "${BASE_OUT_DIR}/flat_discrete"

echo "Running 2/4: rowcol_concat + discrete"
PYTHONPATH=. python toy_scripts/bert_toy_matrix_completion.py \
  "${COMMON_ARGS[@]}" \
  --positional-scheme rowcol_concat \
  --out-dir "${BASE_OUT_DIR}/rowcol_discrete"

echo "Running 3/4: flat + continuous"
PYTHONPATH=. python toy_scripts/bert_toy_matrix_completion.py \
  "${COMMON_ARGS[@]}" \
  --continuous-tokenization \
  --out-dir "${BASE_OUT_DIR}/flat_continuous"

echo "Running 4/4: rowcol_concat + continuous"
PYTHONPATH=. python toy_scripts/bert_toy_matrix_completion.py \
  "${COMMON_ARGS[@]}" \
  --positional-scheme rowcol_concat \
  --continuous-tokenization \
  --out-dir "${BASE_OUT_DIR}/rowcol_continuous"

echo "All 4 runs completed."
