#!/bin/bash
# Sweep num_recurrence at eval for every Recurrent MF run under DOMAIN3-OLD.
# Writes per run: <run-dir>/RECURRENCE_SCALING/{recurrence_scaling.json,recurrence_scaling.png}
#
# Run on a GPU compute node:
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_recurrence_scaling_sweep_domain3_old.sh
#
# Env:
#   RESULTS_ROOT   (default: RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD)
#   CHECKPOINT     (default: last)
#   DEVICE         (default: cuda)
#   RECURRENCE_MAX (default: 8)  — sweep 1..max(RECURRENCE_MAX, trained_r + RECURRENCE_EXTRA)
#   RECURRENCE_EXTRA (default: 0)
#   SKIP_EXISTING  (default: 1)  — skip if recurrence_scaling.json exists
#   CONTINUE_ON_ERROR (default: 1) — keep going after a failed run
#   NO_PLOT        (set to 1 for JSON only)
#   RUN_FILTER     (optional grep pattern on run dir basename, e.g. p0c2r4c0)

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD}"
CHECKPOINT="${CHECKPOINT:-last}"
DEVICE="${DEVICE:-cuda}"
RECURRENCE_MAX="${RECURRENCE_MAX:-8}"
RECURRENCE_EXTRA="${RECURRENCE_EXTRA:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
NO_PLOT="${NO_PLOT:-}"
RUN_FILTER="${RUN_FILTER:-}"

shopt -s nullglob
RUN_DIRS=( "${RESULTS_ROOT}"/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_* )
shopt -u nullglob

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No recurrent runs under ${RESULTS_ROOT}/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*"
    exit 1
fi

echo "============================================================"
echo " Recurrence scaling sweep | ${RESULTS_ROOT}"
echo " CHECKPOINT=${CHECKPOINT}  DEVICE=${DEVICE}"
echo " RECURRENCE_MAX=${RECURRENCE_MAX}  EXTRA=${RECURRENCE_EXTRA}"
echo " Runs found: ${#RUN_DIRS[@]}"
echo "============================================================"

OK=0
SKIP=0
FAIL=0

for RUN_DIR in "${RUN_DIRS[@]}"; do
    NAME="$(basename "$RUN_DIR")"
    if [ -n "$RUN_FILTER" ] && [[ "$NAME" != *"${RUN_FILTER}"* ]]; then
        continue
    fi

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
        echo ">>> SKIP ${NAME}: ${OUT_JSON} exists"
        SKIP=$((SKIP + 1))
        continue
    fi

    RECURRENCES="$(python3 - <<PY
import json
from pathlib import Path
tc = json.loads(Path("${RUN_DIR}/train_config.json").read_text())
trained = int(tc["model"]["num_recurrence"])
end = max(int("${RECURRENCE_MAX}"), trained + int("${RECURRENCE_EXTRA}"))
print(",".join(str(i) for i in range(1, end + 1)))
PY
)"

    echo ""
    echo ">>> ${NAME}  recurrences=${RECURRENCES}"

    export RUN_DIR
    export RECURRENCES
    export CHECKPOINT
    export DEVICE
    export NO_PLOT

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
echo " Per-run: <run>/RECURRENCE_SCALING/"
echo "============================================================"

[ "$FAIL" -eq 0 ]
