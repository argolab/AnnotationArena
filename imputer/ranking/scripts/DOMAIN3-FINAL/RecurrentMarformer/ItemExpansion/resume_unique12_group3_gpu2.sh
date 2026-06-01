#!/bin/bash
# Resume UNIQUE12 sweep runs (group 3/3) from 600 -> 800 epochs on GPU 2.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=2 bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/resume_unique12_group3_gpu2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12}"
export EPOCHS="${EPOCHS:-800}"
export DEVICE="${DEVICE:-cuda}"

RECURRENCE_CONFIGS=(
  "p0c12r1c0   0  12 1   0"
  "p1c1r12c10  1  1  12  10"
)

echo "============================================================"
echo " UNIQUE12 resume group 3/3 | GPU ${CUDA_VISIBLE_DEVICES:-2}"
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
echo "Group 3 complete."
