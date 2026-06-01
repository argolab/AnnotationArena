#!/bin/bash
# UNIQUE12-style sweep with dynamic recurrence — group 2/3 (GPU 1).
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=1 bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_dynr_unique12_group2_gpu1.sh
#
# Models: medium depth / wider core stacks.
# Format: RUN_TAG PRELUDE CORE R_EVAL CODA R_MAX

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-400}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12-DYNR}"

# shellcheck disable=SC2034
RECURRENCE_CONFIGS=(
  "p0c6r2c6   0  6  2  6   8"   # unique=12 eval=18  wide core
  "p3c3r4c6   3  3  4  6  10"   # unique=12 eval=21  symmetric 3+3+4
  "p2c2r4c8   2  2  4  8  10"   # unique=12 eval=18  thin core, mid eval r
)

echo ""
echo "============================================================"
echo " DYNR sweep group 2/3 | GPU ${CUDA_VISIBLE_DEVICES:-1}"
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
echo "Group 2 complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
