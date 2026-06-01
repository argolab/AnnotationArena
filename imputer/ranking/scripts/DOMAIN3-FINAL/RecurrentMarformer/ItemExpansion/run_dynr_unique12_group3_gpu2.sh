#!/bin/bash
# UNIQUE12-style sweep with dynamic recurrence — group 3/3 (GPU 2).
#
#   cd imputer/ranking && export PYTHONPATH=.
#   CUDA_VISIBLE_DEVICES=2 bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_dynr_unique12_group3_gpu2.sh
#
# Models: deep-unroll, thin-core, and flat control (heaviest group).
# Format: RUN_TAG PRELUDE CORE R_EVAL CODA R_MAX

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-400}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12-DYNR}"

# shellcheck disable=SC2034
RECURRENCE_CONFIGS=(
  "p2c2r6c8   2  2  6  8  12"   # unique=12 eval=22  deep eval anchor (UNIQUE12 classic)
  "p1c2r3c7   1  2  3  7  10"   # unique=12 eval=14  thin core, light prelude/coda
  "p0c12r1c0  0  12 1  0   6"   # unique=12 eval=12  flat 12-layer analogue, r=1 eval
)

echo ""
echo "============================================================"
echo " DYNR sweep group 3/3 | GPU ${CUDA_VISIBLE_DEVICES:-2}"
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
echo "Group 3 complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
