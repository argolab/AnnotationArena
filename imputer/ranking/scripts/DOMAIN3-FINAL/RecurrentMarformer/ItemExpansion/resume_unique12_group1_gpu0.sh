#!/bin/bash
# Resume UNIQUE12 sweep runs (group 1/3) from 600 -> 800 epochs on GPU 0.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/resume_unique12_group1_gpu0.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12}"
export EPOCHS="${EPOCHS:-800}"
export DEVICE="${DEVICE:-cuda}"

RECURRENCE_CONFIGS=(
  "p4c4r2c4    4  4  2   4"
  "p0c4r3c8    0  4  3   8"
  "p6c2r3c4    6  2  3   4"
)

echo "============================================================"
echo " UNIQUE12 resume group 1/3 | GPU ${CUDA_VISIBLE_DEVICES:-0}"
echo " EPOCHS=${EPOCHS}  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo " Models: ${#RECURRENCE_CONFIGS[@]}"
echo "============================================================"

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH <<< "$entry"
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_resume_one.sh"
done

echo ""
echo "Group 1 complete."
