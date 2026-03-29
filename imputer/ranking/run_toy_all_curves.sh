#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/xwang397/AnnotationArena/imputer/ranking"
SUFFIX="${1:-run_$(date +%Y%m%d_%H%M%S)}"

AVG_OUT="OUTPUT/toy_average_curves_${SUFFIX}"
SUM_SANITY_OUT="OUTPUT/toy_sum_curves_sanity_${SUFFIX}"
SUM_OUT="OUTPUT/toy_sum_curves_${SUFFIX}"

cd "$BASE_DIR"

# PYTHONPATH=. python toy_scripts/toy_average_task_curves_plot.py --out-dir "$AVG_OUT"
# PYTHONPATH=. python toy_scripts/toy_sum_task_curves_plot.py --sanity-counting --out-dir "$SUM_SANITY_OUT"
PYTHONPATH=. python toy_scripts/toy_sum_task_curves_plot.py --out-dir "$SUM_OUT"