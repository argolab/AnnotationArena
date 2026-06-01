#!/bin/bash
# Train p0c12r1c0 (0,12,1,0) — max_item=300, 1000 epochs.
#
# From ~/AA_new/imputer/ranking:
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_flat_maxitem300_gpu1_p0c12r1c0.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EPOCHS="${EPOCHS:-1000}"
export MAX_ITEM="${MAX_ITEM:-300}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM300}"
export DEVICE="${DEVICE:-cuda}"

export RUN_TAG="p0c12r1c0"
export PRELUDE_DEPTH=0
export NUM_CORE_LAYERS=12
export NUM_RECURRENCE=1
export CODA_DEPTH=0

echo "============================================================"
echo " Flat MAXITEM300 | ${RUN_TAG} (0,12,1,0) -> ${EPOCHS} epochs"
echo " OUTPUT_ROOT=${OUTPUT_ROOT}  MAX_ITEM=${MAX_ITEM}"
echo "============================================================"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_run_one.sh"

echo ""
echo "Done: ${RUN_TAG}"
