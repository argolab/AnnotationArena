#!/usr/bin/env bash
# Four 2×2 matrix-completion runs: oracle param (--show-correct-vector) × multiplication-head.
#
# Configure via environment (defaults suit local runs):
#   LAYERS, HEADS, LR, EMB, D_FF, STEPS, SEED,
#   OUT_ROOT (parent directory under cwd, default OUTPUT/toy_matrix_completion_2x2_ablation)
#   RUN_DIR (optional; default L${LAYERS}_H${HEADS}_Emb${EMB}_lr${LR})
#
# Outputs go to: ${OUT_ROOT}/${RUN_DIR}/{oracle_on_mult_on,oracle_on_mult_off,oracle_off_mult_on,oracle_off_mult_off}/
#
# Usage (from imputer/ranking):
#   bash scripts/test_scripts/run_toy_matrix_completion_2x2_ablation.sh
# Slurm (see submit_sweep_toy_matrix_completion_2x2_ablation.sh):
#   OUT_ROOT=OUTPUT/... LAYERS=4 HEADS=2 LR=1e-4 EMB=16 sbatch_adapt ...

set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS="${STEPS:-2000}"
LR="${LR:-1e-4}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-2}"
EMB="${EMB:-16}"
D_FF="${D_FF:-64}"
SEED="${SEED:-0}"
OUT_ROOT="${OUT_ROOT:-OUTPUT/toy_matrix_completion_2x2_ablation}"
RUN_DIR="${RUN_DIR:-L${LAYERS}_H${HEADS}_Emb${EMB}_lr${LR}}"
OUT_BASE="${OUT_ROOT}/${RUN_DIR}"

BASE=(
  toy_scripts/toy_matrix_completion.py
  --steps "$STEPS"
  --lr "$LR"
  --num-train-graphs 10
  --num-test-graphs 10
  --seed "$SEED"
  --N 2 --M 2 --D 1
  --mask-rate 0.2
  --dropout 0
  --weight-decay 0
  --resample-train-each-step
  --embedding-dim "$EMB"
  --num-layers "$LAYERS"
  --attn-heads "$HEADS"
  --d-ff "$D_FF"
  --num-ffn-layers 1
)

run_one() {
  local name=$1
  shift
  echo "========== ${OUT_BASE}/${name} =========="
  PYTHONPATH=. python "${BASE[@]}" "$@" --out-dir "${OUT_BASE}/${name}"
  echo
}

echo "Run tag: ${RUN_DIR}  (under ${OUT_ROOT})"

# 1) Oracle U/V in row/col param stream + multiplication head
run_one "oracle_on_mult_on" --show-correct-vector --multiplication-head

# 2) Oracle on, multiplication head off
run_one "oracle_on_mult_off" --show-correct-vector

# 3) Oracle off, multiplication head on
run_one "oracle_off_mult_on" --multiplication-head

# 4) Both off
run_one "oracle_off_mult_off"

echo "Done. Curves under ${OUT_BASE}/<variant>/"
