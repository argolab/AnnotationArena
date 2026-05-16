#!/bin/bash
# Sweep with 12 unique blocks (prelude + core + coda = 12), varied unrolling.
#
#   unique_blocks = prelude + core + coda = 12
#   actual_depth  = prelude + core * recurrence + coda
#
# Format per line: RUN_TAG PRELUDE NUM_CORE NUM_RECURRENCE CODA  # comment

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

export EPOCHS="${EPOCHS:-400}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12}"

RECURRENCE_CONFIGS=(
  "p4c4r2c4    4  4  2   4"   # unique=12, actual=16  (example split)
  "p0c12r1c0   0  12 1   0"   # unique=12, actual=12  (flat 12-layer analogue)
  "p0c4r3c8    0  4  3   8"   # unique=12, actual=20
  "p0c6r2c6    0  6  2   6"   # unique=12, actual=18
  "p2c2r6c8    2  2  6   8"   # unique=12, actual=22
  "p3c3r4c6    3  3  4   6"   # unique=12, actual=21
  "p6c2r3c4    6  2  3   4"   # unique=12, actual=16
  "p1c1r12c10  1  1  12  10"   # unique=12, actual=23  (thin core, deep unroll)
)

echo ""
echo "============================================================"
echo " Recurrence sweep unique=12 | DOMAIN3-OLD_Item_T_1000"
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
