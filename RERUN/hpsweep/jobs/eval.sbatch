#!/bin/bash
#SBATCH --job-name=HP-EVAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
OUT="${RERUN_DIR}/hpsweep/out"; mkdir -p "${OUT}"
rerun_run python "${RERUN_DIR}/fig6/grid_eval.py" \
    --run-dir "$(rerun_run_dir "${HP_RUN_ID:-hp-mi300}")" \
    --eval-max-items "${EVAL_MAX_ITEMS:-300}" --checkpoint last \
    --device "${EVAL_DEVICE:-cuda}" \
    -o "${OUT}/${HP_RUN_ID:-hp-mi300}_eval.json"
echo done
