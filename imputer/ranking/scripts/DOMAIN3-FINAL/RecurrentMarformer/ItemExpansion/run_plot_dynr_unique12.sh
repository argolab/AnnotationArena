#!/bin/bash
# Aggregate plots for DYNR UNIQUE12 sweep (CPU ok).
#
# Prerequisites on GPU:
#   bash .../run_eval_sweep_dynr_unique12.sh
#   bash .../run_recurrence_scaling_sweep_dynr_unique12.sh
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_plot_dynr_unique12.sh

set -euo pipefail
export PYTHONPATH=.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12-DYNR}"

python scripts/utils/plot_recurrent_marformer_domain3_sweep.py \
    --results-root "$RESULTS_ROOT" \
    --per-run

echo ""
echo "Plots: PLOTS/TALK/RECURRENT_MARFORMER/${RESULTS_ROOT##*/}/"
echo "  combined_missing_log_loss.png  (training val curve)"
echo "  global_test_log_loss.png       (periodic test eval)"
echo "  global_test_rmse.png"
echo "  recurrence_scaling_log_loss_fullgraph.png (or _maxitemN if not full graph)"
echo "  recurrence_scaling_rmse_fullgraph.png"
