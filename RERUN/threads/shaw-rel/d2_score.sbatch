#!/bin/bash
#SBATCH --job-name=SHAW-D2-SCORE
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --comment=accept_cost
# D2: score every Shaw arm on the test corner and read the paired comparison.
#
# s7_curves.py gives, per run, the corner at the epoch the run's own validation
# block selects, at the corner's own argmin, and at the end of the budget -- so
# the primary read never touches test for selection.
#
# Checkpoints are written every 25 epochs and the budget is 700, so --ckpt-stride 2
# scores 14 of the 28 plus last.ckpt: 50-epoch resolution, which is finer than the
# selector needs and halves ~1260 forward passes.
#
# The eval chunk is 300, matching how these arms were TRAINED. The shipped
# evaluation chunk is 500; using it here would read every arm at a chunk none of
# them trained at, which is a change on top of the one being measured.
set -euo pipefail
RERUN_DIR=/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN
source "${RERUN_DIR}/config/cluster.sh"
# These arms can only be rebuilt by the tree that HAS rel_key/rel_value. The main
# checkout's _reconstruct ignores them, which would silently score every learned
# arm as if it were the bit arm -- a wrong number, not a crash. cluster.sh exports
# PYTHONPATH at source time, so overriding it afterwards wins.
SH=/weka/scratch/jhu/jeisner1/xwang397/Marformer-shaw/imputer/ranking
export DOMAIN3_REPO_ROOT_OVERRIDE="${SH}" PYTHONPATH="${SH}:${PYTHONPATH:-}"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
python - <<'CHK'
import inspect, sys
from imputer.entity_mf import test as t
src = inspect.getsource(t._reconstruct)
missing = [k for k in ("rel_key", "rel_value") if k not in src]
if missing:
    sys.exit(f"FATAL: _reconstruct on PYTHONPATH does not restore {missing}; "
             f"wrong worktree -- every learned arm would score as the bit arm.")
print("ok: _reconstruct restores rel_key and rel_value")
CHK
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
OUT="${RERUN_DIR}/threads/shaw-rel/out_d2.json"
python RERUN/side/s7_curves.py \
    --run-dir "${RERUN_DIR}/runs/shaw-d2-s101" \
    --eval-max-item 300 --ckpt-stride 2 --device cuda -o "${OUT}"
echo
python "${RERUN_DIR}/threads/shaw-rel/d2_analyse.py" --in "${OUT}" --ref bits \
    | tee "${RERUN_DIR}/threads/shaw-rel/report_d2.txt"
echo
python "${RERUN_DIR}/threads/shaw-rel/d2_budget_check.py"
