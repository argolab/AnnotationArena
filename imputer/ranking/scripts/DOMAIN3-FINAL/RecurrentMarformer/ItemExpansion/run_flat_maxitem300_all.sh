#!/bin/bash
# Train flat recurrent configs (0,8,1,0) and (0,12,1,0) with max_item=300 to epoch 1000.
#
# From ~/AA_new/imputer/ranking:
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_flat_maxitem300_all.sh
#
# Parallel (one model per GPU/node):
#   bash .../run_flat_maxitem300_gpu0_p0c8r1c0.sh
#   bash .../run_flat_maxitem300_gpu1_p0c12r1c0.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-1000}"
export MAX_ITEM="${MAX_ITEM:-300}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300}"
export DEVICE="${DEVICE:-cuda}"

# RUN_TAG PRELUDE CORE RECURRENCE CODA
RECURRENCE_CONFIGS=(
  "p0c8r1c0    0  8  1   0"   # unique=8,  actual=8  (flat 8-layer analogue)
  "p0c12r1c0   0  12 1   0"   # unique=12, actual=12 (flat 12-layer analogue)
)

echo ""
echo "============================================================"
echo " Flat recurrent sweep | max_item=${MAX_ITEM} | EPOCHS=${EPOCHS}"
echo " Output: ${OUTPUT_ROOT}"
echo " Models: ${#RECURRENCE_CONFIGS[@]}"
echo "============================================================"

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH <<< "$entry"
    UNIQUE=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS + CODA_DEPTH ))
    ACTUAL=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS * NUM_RECURRENCE + CODA_DEPTH ))
    unset RUN_NAME
    echo ""
    echo ">>> ${RUN_TAG}  unique=${UNIQUE}  actual=${ACTUAL}  max_item=${MAX_ITEM}"
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_run_one.sh"
done

TOTAL=$(( SECONDS - SWEEP_START ))
echo ""
echo "Sweep complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
echo "Runs under ${OUTPUT_ROOT}/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c8r1c0"
echo "         ${OUTPUT_ROOT}/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c12r1c0"
