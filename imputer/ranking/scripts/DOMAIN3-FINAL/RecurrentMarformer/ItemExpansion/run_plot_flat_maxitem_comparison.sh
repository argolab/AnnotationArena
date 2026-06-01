#!/bin/bash
# Plot flat max_item=300 and max_item=500 runs on shared comparison curves.
#
# From ~/AA_new/imputer/ranking:
#   export PYTHONPATH=.
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_plot_flat_maxitem_comparison.sh

set -euo pipefail
export PYTHONPATH=.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../../.."

python scripts/utils/plot_flat_maxitem_comparison.py

echo ""
echo "Plots under PLOTS/TALK/RECURRENT_MARFORMER/DOMAIN3-OLD-FLAT-MAXITEM-COMPARISON/"
