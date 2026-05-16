#!/bin/bash
# Sequentially train several (prelude, core, recurrence, coda) tuples on DOMAIN3-OLD_Item_T_1000.
#
# Each active config has:
#   unique_blocks = prelude + core + coda = 8  (matches flat Marformer param count)
#   actual_depth  = prelude + core * recurrence + coda  (> 8; deeper forward pass)
#
# Format per line: RUN_TAG PRELUDE NUM_CORE NUM_RECURRENCE CODA  # comment

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_START=$SECONDS

# Default training length for this sweep (override: EPOCHS=300 bash run_recurrence_sweep.sh).
export EPOCHS="${EPOCHS:-400}"

# Separate from the first sweep (RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD).
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE8-DEEP}"

# tag  prelude core recurrence coda
RECURRENCE_CONFIGS=(
  # --- Previous sweep (effective_depth = 8; mixed unique block counts) ---
  # "p0c8r1c0  0  8  1  0"   # flat analogue: unique=8, actual=8
  # "p1c2r3c1  1  2  3  1"   # unique=4, actual=8
  # "p0c1r8c0  0  1  8  0"   # unique=1, actual=8
  # "p0c2r4c0  0  2  4  0"   # unique=2, actual=8
  # "p2c2r2c2  2  2  2  2"   # unique=6, actual=8
  # "p2c4r1c2  2  4  1  2"   # unique=8, actual=8
  # "p0c4r2c0  0  4  2  0"   # unique=4, actual=8
  # "p1c1r6c1  1  1  6  1"   # unique=3, actual=8

  # --- New sweep: 8 unique blocks, actual depth > 8 (weight sharing in core) ---
  "p0c4r2c4  0  4  2  4"   # unique=8, actual=12
  "p0c2r4c6  0  2  4  6"   # unique=8, actual=14
  "p1c2r4c5  1  2  4  5"   # unique=8, actual=14
  "p2c2r3c4  2  2  3  4"   # unique=8, actual=12
  "p0c4r3c4  0  4  3  4"   # unique=8, actual=16
  "p1c1r6c6  1  1  6  6"   # unique=8, actual=13
  "p2c4r2c2  2  4  2  2"   # unique=8, actual=12
  "p0c1r8c7  0  1  8  7"   # unique=8, actual=15 (thin shared core, max unroll)
)

echo ""
echo "============================================================"
echo " Recurrence sweep | DOMAIN3-OLD_Item_T_1000"
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
