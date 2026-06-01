#!/bin/bash
# Vary num_recurrence at eval for any trained Recurrent Marformer run.
#
# Usage:
#   RUN_DIR=RESULTS/.../DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p0c2r4c0 \
#   RECURRENCES=1,2,3,4,5,6,7,8 \
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_recurrence_scaling.sh
#
# OOM: run one recurrence per job, e.g.
#   for r in 1 2 3 4 5 6 7 8; do
#     RECURRENCES=$r OUT_DIR=.../RECURRENCE_SCALING/r${r} bash run_recurrence_scaling.sh --no-plot
#   done

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

: "${RUN_DIR:?Set RUN_DIR to a trained run (contains train_config.json and checkpoints/)}"

RECURRENCES="${RECURRENCES:-1,2,3,4,5,6,7,8}"
CHECKPOINT="${CHECKPOINT:-latest}"
DEVICE="${DEVICE:-cuda}"
MAX_ITEM="${MAX_ITEM:-}"
FULL_GRAPH="${FULL_GRAPH:-}"
OUT_DIR="${OUT_DIR:-}"
NO_PLOT="${NO_PLOT:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

ARGS=(
  --run-dir "$RUN_DIR"
  --checkpoint "$CHECKPOINT"
  --recurrences "$RECURRENCES"
  --device "$DEVICE"
)
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$MAX_ITEM" ] && ARGS+=(--max-item "$MAX_ITEM")
[ -n "$FULL_GRAPH" ] && ARGS+=(--full-graph)
[ -n "$NO_PLOT" ] && ARGS+=(--no-plot)

python -u -m imputer.entity_mf.recurrent.recurrence_scaling_eval "${ARGS[@]}"

echo ""
echo "Results: ${OUT_DIR:-${RUN_DIR}/RECURRENCE_SCALING}/"
