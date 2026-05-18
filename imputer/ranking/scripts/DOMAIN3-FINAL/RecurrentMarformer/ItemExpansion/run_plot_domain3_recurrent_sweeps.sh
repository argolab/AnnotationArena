#!/bin/bash
# Plot training / test / recurrence-scaling curves for the three newer DOMAIN3 sweeps.
#
# Prerequisites (GPU node):
#   # UNIQUE12 + UNIQUE8-DEEP need periodic test eval first:
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_sweep_unique12.sh
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_sweep_unique8_deep.sh
#
#   # Recurrence-at-eval scaling (all three sweeps):
#   for ROOT in DOMAIN3-OLD-UNIQUE12 DOMAIN3-OLD-P0C1RX DOMAIN3-OLD-UNIQUE8-DEEP; do
#     RESULTS_ROOT="RESULTS/RECURRENT_MARFORMER/${ROOT}" \
#       bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_recurrence_scaling_sweep_domain3_old.sh
#   done
#
# Then (CPU ok):
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_plot_domain3_recurrent_sweeps.sh

set -euo pipefail
export PYTHONPATH=.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

python scripts/utils/plot_recurrent_marformer_domain3_sweep.py --all-sweeps --per-run

echo ""
echo "Plots under PLOTS/TALK/RECURRENT_MARFORMER/<sweep>/"
echo "Training overlay also copied to each RESULTS/RECURRENT_MARFORMER/<sweep>/combined_missing_log_loss.png"
