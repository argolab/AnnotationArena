#!/bin/bash
# Same as run_dynr_unique12_group1_gpu0.sh but MAX_ITEM=300 (fuller graphs per step).
# Results go to a separate output root — does not touch DOMAIN3-OLD-UNIQUE12-DYNR (max_item=100).
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_dynr_unique12_group1_gpu0_maxitem300.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-400}"
export MAX_ITEM="${MAX_ITEM:-300}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12-DYNR-MAXITEM300}"
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-_DYNR_M300}"

# shellcheck disable=SC2034
RECURRENCE_CONFIGS=(
  "p6c2r3c4   6  2  3  4   8"
  "p4c4r2c4   4  4  2  4   8"
  "p0c4r3c8   0  4  3  8  10"
)

echo ""
echo "============================================================"
echo " DYNR sweep group 1 | MAX_ITEM=${MAX_ITEM} | GPU ${CUDA_VISIBLE_DEVICES:-0}"
echo " EPOCHS=${EPOCHS}  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo " Models: ${#RECURRENCE_CONFIGS[@]}"
echo "============================================================"

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH RECURRENCE_MAX <<< "$entry"
    UNIQUE=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS + CODA_DEPTH ))
    EVAL_DEPTH=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS * NUM_RECURRENCE + CODA_DEPTH ))
    export RUN_NAME="DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_${RUN_TAG}${RUN_NAME_SUFFIX}"
    echo ""
    echo ">>> ${RUN_TAG}  unique=${UNIQUE} eval_depth=${EVAL_DEPTH} r_train_max=${RECURRENCE_MAX} max_item=${MAX_ITEM}"
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH RECURRENCE_MAX
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_run_one_dynr.sh"
done

TOTAL=$(( SECONDS - SWEEP_START ))
echo ""
echo "Group 1 (MAX_ITEM=${MAX_ITEM}) complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
