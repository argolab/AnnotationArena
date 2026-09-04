#!/bin/bash
#SBATCH --job-name=RESID-REFIT-EVAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail
source "${RERUN_DIR}/config/cluster.sh"; rerun_activate_env
mkdir -p "${RERUN_DIR}/reports/build"
rerun_run python "${RERUN_DIR}/fig6/grid_eval.py" \
  --run-dir "$(rerun_run_dir resid-refit-01)" --eval-max-items 300 \
  --checkpoint last --device cpu -o "${RERUN_DIR}/reports/build/resid_refit_eval.json"
