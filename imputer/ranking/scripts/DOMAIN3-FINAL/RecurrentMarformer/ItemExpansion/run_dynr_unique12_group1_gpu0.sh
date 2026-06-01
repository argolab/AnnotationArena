#!/bin/bash
# UNIQUE12-style sweep with dynamic recurrence — group 1/3 (GPU 0).
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_dynr_unique12_group1_gpu0.sh
#
# Models: balanced prelude/core splits, eval r in {2,3}, train r up to 8–10.
# Format: RUN_TAG PRELUDE CORE R_EVAL CODA R_MAX

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-400}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12-DYNR}"

# shellcheck disable=SC2034
RECURRENCE_CONFIGS=(
  "p6c2r3c4   6  2  3  4   8"   # unique=12 eval=16  balanced (plan default)
  "p4c4r2c4   4  4  2  4   8"   # unique=12 eval=16  thick core, shallow eval r
  "p0c4r3c8   0  4  3  8  10"   # unique=12 eval=20  wide core, long coda
)

echo ""
echo "============================================================"
echo " DYNR sweep group 1/3 | GPU ${CUDA_VISIBLE_DEVICES:-0}"
echo " EPOCHS=${EPOCHS}  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo " Models: ${#RECURRENCE_CONFIGS[@]}"
echo "============================================================"

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH RECURRENCE_MAX <<< "$entry"
    UNIQUE=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS + CODA_DEPTH ))
    EVAL_DEPTH=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS * NUM_RECURRENCE + CODA_DEPTH ))
    echo ""
    echo ">>> ${RUN_TAG}_DYNR  unique=${UNIQUE} eval_depth=${EVAL_DEPTH} r_train_max=${RECURRENCE_MAX}"
    unset RUN_NAME
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH RECURRENCE_MAX
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_run_one_dynr.sh"
done

TOTAL=$(( SECONDS - SWEEP_START ))
echo ""
echo "Group 1 complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
