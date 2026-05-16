#!/bin/bash
# Sweep (0, 1, x, 0) on DOMAIN3-OLD_Item_T_1000: one shared core block, unrolled x times.
#
#   unique_blocks = 0 + 1 + 0 = 1
#   actual_depth  = x  (recurrence ∈ {6, 8, 10, 12, 14, 16})
#
# Format per line: RUN_TAG PRELUDE NUM_CORE NUM_RECURRENCE CODA  # comment

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-400}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-P0C1RX}"

RECURRENCE_CONFIGS=(
  "p0c1r6c0   0  1  6   0"   # unique=1, actual=6
  "p0c1r8c0   0  1  8   0"   # unique=1, actual=8
  "p0c1r10c0  0  1  10  0"   # unique=1, actual=10
  "p0c1r12c0  0  1  12  0"   # unique=1, actual=12
  "p0c1r14c0  0  1  14  0"   # unique=1, actual=14
  "p0c1r16c0  0  1  16  0"   # unique=1, actual=16
)

echo ""
echo "============================================================"
echo " Recurrence sweep p0c1r{x}c0 | DOMAIN3-OLD_Item_T_1000"
echo " EPOCHS        : ${EPOCHS}"
echo " Configs       : ${#RECURRENCE_CONFIGS[@]}"
echo " Output root   : ${OUTPUT_ROOT}"
echo "============================================================"

for entry in "${RECURRENCE_CONFIGS[@]}"; do
    read -r RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH _rest <<< "$entry"
    UNIQUE=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS + CODA_DEPTH ))
    ACTUAL=$(( PRELUDE_DEPTH + NUM_CORE_LAYERS * NUM_RECURRENCE + CODA_DEPTH ))
    echo ""
    echo ">>> Starting ${RUN_TAG} (prelude=${PRELUDE_DEPTH} core=${NUM_CORE_LAYERS} rec=${NUM_RECURRENCE} coda=${CODA_DEPTH} unique=${UNIQUE} actual=${ACTUAL})"
    export RUN_TAG PRELUDE_DEPTH NUM_CORE_LAYERS NUM_RECURRENCE CODA_DEPTH
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/_run_one.sh"
done

TOTAL=$(( SECONDS - SWEEP_START ))
echo ""
echo "============================================================"
echo " Sweep complete in $(( TOTAL / 60 ))m $(( TOTAL % 60 ))s"
echo " Runs under ${OUTPUT_ROOT}/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*"
echo "============================================================"
