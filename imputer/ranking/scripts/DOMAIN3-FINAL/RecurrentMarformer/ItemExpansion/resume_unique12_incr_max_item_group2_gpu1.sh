#!/bin/bash
# Resume UNIQUE12 copies (INCR_MAX_ITEM) with max_item=200 to epoch 1500 — group 2/3.
#
# From ~/AA_new/imputer/ranking:
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/resume_unique12_incr_max_item_group2_gpu1.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_INCR_MAX_ITEM}"
export MAX_ITEM="${MAX_ITEM:-200}"
export EPOCHS="${EPOCHS:-1500}"
export DEVICE="${DEVICE:-cuda}"

RECURRENCE_CONFIGS=(
  "p0c6r2c6    0  6  2   6"
  "p3c3r4c6    3  3  4   6"
  "p2c2r6c8    2  2  6   8"
)

echo "============================================================"
echo " UNIQUE12 INCR_MAX_ITEM resume group 2/3"
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
echo "Group 2 (INCR_MAX_ITEM) complete."
