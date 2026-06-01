#!/bin/bash
# Resume UNIQUE12 copies (MAXITEM150) with max_item=150 to epoch 1000 — group 1/3.
# Writes only under DOMAIN3-OLD-UNIQUE12_MAXITEM150 (not DOMAIN3-OLD-UNIQUE12).
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/.../copy_unique12_runs_to_maxitem150.sh   # once
#   CUDA_VISIBLE_DEVICES=0 bash scripts/.../resume_unique12_maxitem150_group1_gpu0.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_MAXITEM150}"
export MAX_ITEM="${MAX_ITEM:-150}"
export EPOCHS="${EPOCHS:-1000}"
export DEVICE="${DEVICE:-cuda}"

RECURRENCE_CONFIGS=(
  "p4c4r2c4    4  4  2   4"
  "p0c4r3c8    0  4  3   8"
  "p6c2r3c4    6  2  3   4"
)

echo "============================================================"
echo " UNIQUE12 MAXITEM150 resume group 1/3 | GPU ${CUDA_VISIBLE_DEVICES:-0}"
echo " OUTPUT_ROOT=${OUTPUT_ROOT}  MAX_ITEM=${MAX_ITEM}  EPOCHS=${EPOCHS}"
echo " Models: ${#RECURRENCE_CONFIGS[@]}"
echo "============================================================"

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH <<< "$entry"
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_resume_one.sh"
done

echo ""
echo "Group 1 (MAXITEM150) complete."
