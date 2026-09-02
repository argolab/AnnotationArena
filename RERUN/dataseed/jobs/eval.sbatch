#!/bin/bash
#SBATCH --job-name=DSEED-EVAL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
set -euo pipefail
# Score every data-seed checkpoint on its own corner and on its own training
# cells, reusing fig6/grid_eval.py unchanged so the numbers are produced by
# exactly the code that produced the grid's.
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
: "${SEEDS:?}"
OUT="${RERUN_DIR}/dataseed/out"
mkdir -p "${OUT}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
for s in "${SEED_LIST[@]}"; do
    echo "=========== data seed ${s} ==========="
    rerun_run python "${RERUN_DIR}/fig6/grid_eval.py" \
        --run-dir "$(rerun_run_dir "dataseed-s${s}")" \
        --eval-max-items "${EVAL_MAX_ITEMS:-300}" \
        --checkpoint last \
        -o "${OUT}/eval_s${s}.json"
done
echo "done"
