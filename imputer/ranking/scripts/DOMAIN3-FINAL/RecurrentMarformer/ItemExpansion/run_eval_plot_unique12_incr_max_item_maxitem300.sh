#!/bin/bash
# INCR_MAX_ITEM sweep: test + recurrence eval at max_item=300, plots in a separate dir.
#
# From ~/AA_new/imputer/ranking (already the ranking root):
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_eval_plot_unique12_incr_max_item_maxitem300.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RESULTS_ROOT="${RESULTS_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_INCR_MAX_ITEM}"
export MAX_ITEM="${MAX_ITEM:-300}"
export RECURRENCE_MAX="${RECURRENCE_MAX:-12}"
export SKIP_EXISTING="${SKIP_EXISTING:-0}"
export RESUME_EPOCH="${RESUME_EPOCH:-600}"
export PHASE_NOTE="${PHASE_NOTE:-train max_item=200; test eval max_item=300 (epochs >600)}"
export PLOT_SUFFIX="${PLOT_SUFFIX:-_MAXITEM300}"

bash "${SCRIPT_DIR}/run_eval_plot_recurrent_sweep.sh"
