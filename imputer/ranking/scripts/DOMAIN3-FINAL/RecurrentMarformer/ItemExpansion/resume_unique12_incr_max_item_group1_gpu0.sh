#!/bin/bash
# Resume UNIQUE12 copies (INCR_MAX_ITEM) with max_item=200 to epoch 1500 — group 1/3.
# Writes only under DOMAIN3-OLD-UNIQUE12_INCR_MAX_ITEM (not DOMAIN3-OLD-UNIQUE12).
#
# From ~/AA_new/imputer/ranking (this directory is the ranking root; do not cd imputer/ranking again):
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/resume_unique12_incr_max_item_group1_gpu0.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_INCR_MAX_ITEM}"
export MAX_ITEM="${MAX_ITEM:-200}"
export EPOCHS="${EPOCHS:-1500}"
export DEVICE="${DEVICE:-cuda}"

RECURRENCE_CONFIGS=(
  "p4c4r2c4    4  4  2   4"
  "p0c4r3c8    0  4  3   8"
  "p6c2r3c4    6  2  3   4"
)

echo "============================================================"
echo " UNIQUE12 INCR_MAX_ITEM resume group 1/3"
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
echo "Group 1 (INCR_MAX_ITEM) complete."
