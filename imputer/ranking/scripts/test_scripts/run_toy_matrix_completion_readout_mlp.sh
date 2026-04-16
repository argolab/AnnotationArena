#!/usr/bin/env bash
# Run one toy matrix-completion readout-MLP configuration.
#
# Usage (from imputer/ranking):
#   bash scripts/test_scripts/run_toy_matrix_completion_readout_mlp.sh
#
# Common overrides:
#   STEPS=10000 READOUT_MLP_LAYER=2 READOUT_MLP_DIM=128 \
#   OUT_ROOT=OUTPUT/mc_readout_mlp bash scripts/test_scripts/run_toy_matrix_completion_readout_mlp.sh
#
# Notes:
# - This script intentionally does NOT pass --show-correct-vector.
# - This script intentionally does NOT pass --multiplication-head.

set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS="${STEPS:-10000}"
SEED="${SEED:-42}"
READOUT_MLP_LAYER="${READOUT_MLP_LAYER:-0}"
READOUT_MLP_DIM="${READOUT_MLP_DIM:-0}"
OUT_ROOT="${OUT_ROOT:-OUTPUT/mc_readout_mlp}"
RUN_TAG="${RUN_TAG:-L${READOUT_MLP_LAYER}_D${READOUT_MLP_DIM}}"
OUT_DIR="${OUT_ROOT}/${RUN_TAG}"

echo "Running readout-MLP config: layer=${READOUT_MLP_LAYER}, dim=${READOUT_MLP_DIM}"
echo "Output dir: ${OUT_DIR}"

PYTHONPATH=. python toy_scripts/toy_matrix_completion.py \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --readout-mlp-layer "${READOUT_MLP_LAYER}" \
  --readout-mlp-dim "${READOUT_MLP_DIM}" \
  --out-dir "${OUT_DIR}"

