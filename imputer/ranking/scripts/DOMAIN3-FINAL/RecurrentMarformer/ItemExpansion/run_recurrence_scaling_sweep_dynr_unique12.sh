#!/bin/bash
# Recurrence-at-eval sweep for all DYNR UNIQUE12 runs.
# Sweeps r = 1 .. max(RECURRENCE_MAX, trained_r, recurrence_max from train_config).
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_recurrence_scaling_sweep_dynr_unique12.sh
#
# Per run: <run-dir>/RECURRENCE_SCALING/{recurrence_scaling.json, recurrence_scaling.png}
# Uses --full-graph style eval (max_item=None) inside recurrence_scaling_eval.

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12-DYNR}"
GLOB="DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*_DYNR"
CHECKPOINT="${CHECKPOINT:-last}"
DEVICE="${DEVICE:-cuda}"
RECURRENCE_MAX="${RECURRENCE_MAX:-12}"
RECURRENCE_EXTRA="${RECURRENCE_EXTRA:-0}"
MAX_ITEM="${MAX_ITEM:-}"
FULL_GRAPH="${FULL_GRAPH:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
NO_PLOT="${NO_PLOT:-}"

shopt -s nullglob
RUN_DIRS=( "${RESULTS_ROOT}"/${GLOB} )
shopt -u nullglob

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No runs under ${RESULTS_ROOT}/${GLOB}"
    exit 1
fi

echo "============================================================"
echo " DYNR recurrence scaling | ${RESULTS_ROOT}"
echo " CHECKPOINT=${CHECKPOINT}  cap RECURRENCE_MAX=${RECURRENCE_MAX}  FULL_GRAPH=${FULL_GRAPH:-0}  MAX_ITEM=${MAX_ITEM:-train_config}"
echo " Runs: ${#RUN_DIRS[@]}"
echo "============================================================"

OK=0
SKIP=0
FAIL=0

for RUN_DIR in "${RUN_DIRS[@]}"; do
    NAME="$(basename "$RUN_DIR")"

    if [ ! -f "${RUN_DIR}/train_config.json" ]; then
        echo ">>> SKIP ${NAME}: no train_config.json"
        SKIP=$((SKIP + 1))
        continue
    fi

    CKPT_DIR="${RUN_DIR}/checkpoints"
    if [ "$CHECKPOINT" = "last" ] && [ ! -f "${CKPT_DIR}/last.ckpt" ]; then
        echo ">>> SKIP ${NAME}: no checkpoints/last.ckpt"
        SKIP=$((SKIP + 1))
        continue
    fi

    OUT_JSON="${RUN_DIR}/RECURRENCE_SCALING/recurrence_scaling.json"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$OUT_JSON" ]; then
        echo ">>> SKIP ${NAME}: exists"
        SKIP=$((SKIP + 1))
        continue
    fi

    RECURRENCES="$(python3 - <<PY
import json
from pathlib import Path
tc = json.loads(Path("${RUN_DIR}/train_config.json").read_text())
m = tc["model"]
trained = int(m["num_recurrence"])
r_max_train = int(m.get("recurrence_max", trained))
end = max(int("${RECURRENCE_MAX}"), trained + int("${RECURRENCE_EXTRA}"), r_max_train)
print(",".join(str(i) for i in range(1, end + 1)))
PY
)"

    echo ""
    echo ">>> ${NAME}  recurrences=${RECURRENCES}"

    export RUN_DIR RECURRENCES CHECKPOINT DEVICE MAX_ITEM FULL_GRAPH NO_PLOT

    if bash "${SCRIPT_DIR}/run_recurrence_scaling.sh"; then
        OK=$((OK + 1))
    else
        echo ">>> FAILED ${NAME}"
        FAIL=$((FAIL + 1))
        if [ "$CONTINUE_ON_ERROR" != "1" ]; then
            exit 1
        fi
    fi
done

echo ""
echo "============================================================"
echo " Done: ok=${OK}  skipped=${SKIP}  failed=${FAIL}"
echo "============================================================"
[ "$FAIL" -eq 0 ]
