#!/bin/bash
# Test-eval all runs in the three newer DOMAIN3 sweeps with MAX_ITEM=300, then refresh plots.
#
# GPU node:
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_three_sweeps_maxitem300.sh
#
# Split across 2 GPUs (run in parallel in separate terminals):
#   MAX_ITEM=300 CUDA_VISIBLE_DEVICES=0 bash .../run_eval_sweep_unique12.sh
#   MAX_ITEM=300 CUDA_VISIBLE_DEVICES=1 bash .../run_eval_sweep_unique8_deep.sh
#   MAX_ITEM=300 CUDA_VISIBLE_DEVICES=0 bash .../run_eval_sweep_p0c1rx.sh
#
# Plots only (CPU ok):
#   SKIP_EVAL=1 bash .../run_eval_three_sweeps_maxitem300.sh

set -euo pipefail
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

export MAX_ITEM="${MAX_ITEM:-300}"
export DEVICE="${DEVICE:-cuda}"

echo "============================================================"
echo " Recurrent MF test eval | MAX_ITEM=${MAX_ITEM}  DEVICE=${DEVICE}"
echo "============================================================"

if [ "${SKIP_EVAL:-0}" != "1" ]; then
    echo ""
    echo "=== DOMAIN3-OLD-UNIQUE12 ==="
    bash "${SCRIPT_DIR}/run_eval_sweep_unique12.sh"

    echo ""
    echo "=== DOMAIN3-OLD-UNIQUE8-DEEP ==="
    bash "${SCRIPT_DIR}/run_eval_sweep_unique8_deep.sh"

    echo ""
    echo "=== DOMAIN3-OLD-P0C1RX ==="
    bash "${SCRIPT_DIR}/run_eval_sweep_p0c1rx.sh"
fi

echo ""
echo "=== Plots -> PLOTS/TALK/RECURRENT_MARFORMER/ ==="
bash "${SCRIPT_DIR}/run_plot_domain3_recurrent_sweeps.sh"

echo ""
echo "Done."
