#!/bin/bash
# GPU: periodic test eval + recurrence scaling for one Recurrent MF sweep root.
# CPU: aggregate plots (optionally mark resume phase after epoch 600).
#
# Required env:
#   RESULTS_ROOT   e.g. RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_MAXITEM150
#
# Optional env:
#   MAX_ITEM           (e.g. 300 -> TEST_RESULTS_MAXITEM300 + separate plot dir)
#   PLOT_SUFFIX        (default: _MAXITEM300 when MAX_ITEM=300, else empty)
#   RECURRENCE_MAX     (default 12 for UNIQUE12-style sweeps)
#   SKIP_EXISTING      (default 0 for recurrence JSON — set 1 to skip existing)
#   DEVICE             (default cuda)
#   RESUME_EPOCH       (default empty; e.g. 600 for copied-checkpoint resumes)
#   PHASE_NOTE         (default empty; e.g. "max_item=150 (epochs >600)")
#   SKIP_EVAL=1        (plot only, skip test + recurrence eval)
#   SKIP_RECURRENCE=1  (run periodic test eval, but skip recurrence-scaling eval)
#   SKIP_PLOT=1        (eval only)
#   PER_RUN_PLOTS=1    (pass --per-run to plot script)
#
# Example:
#   cd imputer/ranking && export PYTHONPATH=.
#   RESULTS_ROOT=RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_MAXITEM150 \
#   RESUME_EPOCH=600 PHASE_NOTE='max_item=150 (epochs >600)' \
#     bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_plot_recurrent_sweep.sh

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

: "${RESULTS_ROOT:?Set RESULTS_ROOT (e.g. RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_MAXITEM150)}"

RECURRENCE_MAX="${RECURRENCE_MAX:-12}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_ITEM="${MAX_ITEM:-}"
FULL_GRAPH="${FULL_GRAPH:-}"
RESUME_EPOCH="${RESUME_EPOCH:-}"
PHASE_NOTE="${PHASE_NOTE:-}"
SKIP_EVAL="${SKIP_EVAL:-}"
SKIP_RECURRENCE="${SKIP_RECURRENCE:-}"
SKIP_PLOT="${SKIP_PLOT:-}"
PER_RUN_PLOTS="${PER_RUN_PLOTS:-1}"

SWEEP_NAME="${RESULTS_ROOT##*/}"

EVAL_SUBDIRS="$(python3 -c "
from imputer.entity_mf.recurrent.eval_paths import (
    test_results_dir_name,
    recurrence_scaling_dir_name,
    max_item_plot_tag,
)
mi = int('${MAX_ITEM}') if '${MAX_ITEM}' else None
print(test_results_dir_name(mi, full_graph=bool('${FULL_GRAPH}')))
print(recurrence_scaling_dir_name(mi, full_graph=bool('${FULL_GRAPH}')))
print(max_item_plot_tag(mi))
")"
TEST_RESULTS_SUBDIR="$(echo "$EVAL_SUBDIRS" | sed -n '1p')"
SCALING_SUBDIR="$(echo "$EVAL_SUBDIRS" | sed -n '2p')"
MI_TAG="$(echo "$EVAL_SUBDIRS" | sed -n '3p')"
PLOT_SUFFIX="${PLOT_SUFFIX:-}"
if [ -z "$PLOT_SUFFIX" ] && [ -n "$MAX_ITEM" ]; then
    PLOT_SUFFIX="_MAXITEM${MAX_ITEM}"
elif [ -z "$PLOT_SUFFIX" ] && [ -n "$FULL_GRAPH" ]; then
    PLOT_SUFFIX="_FULLGRAPH"
fi
PLOT_OUT_DIR="${PLOT_OUT_DIR:-PLOTS/TALK/RECURRENT_MARFORMER/${SWEEP_NAME}${PLOT_SUFFIX}}"

if [ -z "$SKIP_EVAL" ]; then
    echo "============================================================"
    echo " Test eval | ${SWEEP_NAME}"
    echo " RESULTS_ROOT=${RESULTS_ROOT}  MAX_ITEM=${MAX_ITEM:-train_config}"
    echo "============================================================"
    RESULTS_ROOT="$RESULTS_ROOT" DEVICE="$DEVICE" MAX_ITEM="${MAX_ITEM:-}" FULL_GRAPH="$FULL_GRAPH" \
        bash "${SCRIPT_DIR}/run_eval_sweep_unique12.sh"

    if [ -z "$SKIP_RECURRENCE" ]; then
        echo ""
        echo "============================================================"
        echo " Recurrence scaling | ${SWEEP_NAME}"
        echo " RECURRENCE_MAX=${RECURRENCE_MAX}  SKIP_EXISTING=${SKIP_EXISTING}"
        echo "============================================================"
        RESULTS_ROOT="$RESULTS_ROOT" DEVICE="$DEVICE" MAX_ITEM="${MAX_ITEM:-}" FULL_GRAPH="$FULL_GRAPH" \
            RECURRENCE_MAX="$RECURRENCE_MAX" SKIP_EXISTING="$SKIP_EXISTING" \
            bash "${SCRIPT_DIR}/run_recurrence_scaling_sweep_domain3_old.sh"
    else
        echo ""
        echo "============================================================"
        echo " Skip recurrence scaling | ${SWEEP_NAME}"
        echo " SKIP_RECURRENCE=${SKIP_RECURRENCE}"
        echo "============================================================"
    fi
fi

if [ -z "$SKIP_PLOT" ]; then
    echo ""
    echo "============================================================"
    echo " Plot | ${SWEEP_NAME}"
    echo " RESUME_EPOCH=${RESUME_EPOCH:-none}  PHASE_NOTE=${PHASE_NOTE:-none}"
    echo "============================================================"
    PLOT_ARGS=(
        --results-root "$RESULTS_ROOT"
        --output-dir "$PLOT_OUT_DIR"
        --test-results-subdir "$TEST_RESULTS_SUBDIR"
        --scaling-subdir "$SCALING_SUBDIR"
    )
    if [ -n "$MAX_ITEM" ]; then
        PLOT_ARGS+=(--eval-max-item "$MAX_ITEM")
    fi
    [ "$PER_RUN_PLOTS" = "1" ] && PLOT_ARGS+=(--per-run)
    [ -n "$RESUME_EPOCH" ] && PLOT_ARGS+=(--resume-epoch "$RESUME_EPOCH")
    [ -n "$PHASE_NOTE" ] && PLOT_ARGS+=(--phase-note "$PHASE_NOTE")
    python scripts/utils/plot_recurrent_marformer_domain3_sweep.py "${PLOT_ARGS[@]}"
fi

echo ""
echo "Done: ${SWEEP_NAME}"
echo "  Test/scaling under each run: ${TEST_RESULTS_SUBDIR}/  ${SCALING_SUBDIR}/"
echo "  Plots -> ${PLOT_OUT_DIR}/"
echo "  Training overlay -> ${RESULTS_ROOT}/combined_missing_log_loss.png"
