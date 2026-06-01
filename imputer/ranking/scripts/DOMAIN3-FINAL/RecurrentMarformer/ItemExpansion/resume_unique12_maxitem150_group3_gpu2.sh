#!/bin/bash
# Resume UNIQUE12 copies (MAXITEM150) with max_item=150 to epoch 1000 — group 3/3.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=2 bash scripts/.../resume_unique12_maxitem150_group3_gpu2.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_MAXITEM150}"
export MAX_ITEM="${MAX_ITEM:-150}"
export EPOCHS="${EPOCHS:-1000}"
export DEVICE="${DEVICE:-cuda}"

RECURRENCE_CONFIGS=(
  "p0c12r1c0   0  12 1   0"
  "p1c1r12c10  1  1  12  10"
)

echo "============================================================"
echo " UNIQUE12 MAXITEM150 resume group 3/3 | GPU ${CUDA_VISIBLE_DEVICES:-2}"
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
echo "Group 3 (MAXITEM150) complete."
