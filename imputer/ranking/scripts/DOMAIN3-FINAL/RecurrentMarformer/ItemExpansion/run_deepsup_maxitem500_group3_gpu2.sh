#!/bin/bash
# Deep-supervision sweep group 3/3 (GPU 2): r in {12, 14, 16}, coda=0, max_item=500.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=2 bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_deepsup_maxitem500_group3_gpu2.sh
#
# Format: RUN_TAG PRELUDE CORE R

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-400}"
export MAX_ITEM="${MAX_ITEM:-500}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-DEEPSUP-MAXITEM500}"
export RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-_DS_M500}"

# shellcheck disable=SC2034
RECURRENCE_CONFIGS=(
  "p10c2r12c0 10  2 12"   # unique=12  eff=34
  "p6c2r14c0   6  2 14"   # unique=8   eff=34
  "p8c2r16c0   8  2 16"   # unique=10  eff=40
)

echo ""
echo "============================================================"
echo " DEEPSUP sweep group 3/3 | MAX_ITEM=${MAX_ITEM} | GPU ${CUDA_VISIBLE_DEVICES:-2}"
echo " EPOCHS=${EPOCHS}  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo " Models: ${#RECURRENCE_CONFIGS[@]}"
echo "============================================================"

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE <<< "$entry"
    UNIQUE=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS ))
    EFF_DEPTH=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS * NUM_RECURRENCE ))
    echo ""
    echo ">>> ${RUN_TAG}  unique=${UNIQUE} eff_depth=${EFF_DEPTH} r=${NUM_RECURRENCE}"
    unset RUN_NAME
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_run_one_deepsup.sh"
done

TOTAL=$(( SECONDS - SWEEP_START ))
echo ""
echo "Group 3 complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
