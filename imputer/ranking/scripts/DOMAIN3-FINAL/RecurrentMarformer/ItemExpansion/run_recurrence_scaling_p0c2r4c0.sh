#!/bin/bash
# p0c2r4c0 (0,2,4,0) last.ckpt: vary num_recurrence at eval.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_DIR="${RUN_DIR:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c2r4c0}"
export RECURRENCES="${RECURRENCES:-1,2,3,4,5,6,7,8}"
export CHECKPOINT="${CHECKPOINT:-last}"
export DEVICE="${DEVICE:-cuda}"

exec bash "${SCRIPT_DIR}/run_recurrence_scaling.sh"
