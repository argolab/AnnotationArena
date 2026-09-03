#!/bin/bash
#SBATCH --job-name=DSEED-EVALCPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
set -euo pipefail
# CPU fallback. The corner is 414 cells and the graphs are a few thousand
# tokens, so this is minutes of CPU -- worth having when every GPU partition is
# saturated and the GPU copy is sitting behind a priority queue.
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
OUT="${RERUN_DIR}/dataseed/out"; mkdir -p "${OUT}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
for s in "${SEED_LIST[@]}"; do
    rerun_run python "${RERUN_DIR}/fig6/grid_eval.py" \
        --run-dir "$(rerun_run_dir "dataseed-s${s}")" \
        --eval-max-items 300 --checkpoint last --device cpu \
        -o "${OUT}/eval_s${s}.json"
done
echo done
