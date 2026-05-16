#!/bin/bash
# Single default recurrence tuple on DATA/DOMAIN3-OLD_Item_T_1000
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_START=$SECONDS

RUN_TAG="p1c2r3c1"
PRELUDE_DEPTH=1
NUM_CORE_LAYERS=2
NUM_RECURRENCE=3
CODA_DEPTH=1
export EPOCHS="${EPOCHS:-400}"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_run_one.sh"

echo " Done in $(( (SECONDS - SCRIPT_START) / 60 ))m"
