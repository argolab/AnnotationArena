#!/bin/bash
# DOMAIN3-OLD-UNIQUE12: resumed 600 -> 800 epochs, max_item=100 unchanged.
#
#   cd imputer/ranking && export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_plot_unique12_resumed800.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12}"
export RECURRENCE_MAX="${RECURRENCE_MAX:-12}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export RESUME_EPOCH="${RESUME_EPOCH:-600}"
export PHASE_NOTE="${PHASE_NOTE:-max_item=100 (epochs >600)}"

bash "${SCRIPT_DIR}/run_eval_plot_recurrent_sweep.sh"
