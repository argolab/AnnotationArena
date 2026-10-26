#!/bin/bash
#SBATCH --job-name=B11-EVAL
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --comment=accept_cost
# C2/C5: score every block-projection arm on the corner.
#
# The four arms differ only in whether the trailing d->d projection and the
# outer residual add are present, so the comparison is paired on
# (draw, model seed, K) and read against proj_keep_add, the shipped block.
#
# All four arms trained at max_item=300, so scoring at a common evaluation
# chunk is fair to all of them. 500 is the shipped evaluation setting and runs
# first; 300 is the training-matched read and is a control on it.
#
#   B11_RUN    = run id (proj-c2-s101 | proj-c2-s105)
#   B11_CHUNKS = comma-separated evaluation chunks
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
# These arms were TRAINED from the block-projection worktree, which is the only
# tree that has `value_codec` at all. The patches checkout cannot
# rebuild a proj_drop_rot model -- it builds the shipped block and the load
# fails on missing proj_out keys. cluster.sh exports PYTHONPATH at source time,
# so overriding it here wins.
BP=/weka/scratch/jhu/jeisner1/xwang397/Marformer-codec/imputer/ranking
export DOMAIN3_REPO_ROOT_OVERRIDE="${BP}" PYTHONPATH="${BP}:${PYTHONPATH:-}"
rerun_activate_env
cd "$(dirname "${RERUN_DIR}")"
# Fail now, not 35 seconds into the first checkpoint, if the tree on the path
# cannot restore the flag the arms differ in.
python - <<'CHK'
import inspect, sys
from imputer.entity_mf import test as t
if "value_codec" not in inspect.getsource(t._reconstruct):
    sys.exit("FATAL: _reconstruct on PYTHONPATH does not restore "
             "value_codec; wrong worktree.")
print("ok: _reconstruct restores value_codec")
CHK
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
OUT_DIR="${RERUN_DIR}/side/out"
mkdir -p "${OUT_DIR}"
IFS=',' read -r -a CHUNKS <<< "${B11_CHUNKS:-500,300}"
for c in "${CHUNKS[@]}"; do
    echo "=== ${B11_RUN} at evaluation chunk ${c} ($(date +%H:%M:%S)) ==="
    # analyse_arms.py reads side/out/arms_<run>.json, so the shipped chunk
    # writes that exact name and the control gets a suffixed one.
    if [[ "${c}" == "500" ]]; then
        out="${OUT_DIR}/arms_${B11_RUN}.json"
    else
        out="${OUT_DIR}/arms_${B11_RUN}_mi${c}.json"
    fi
    python RERUN/side/s7_curves.py \
        --run-dir "${RERUN_DIR}/runs/${B11_RUN}" \
        --eval-max-item "${c}" \
        --device cpu \
        -o "${out}"
done
echo "done"
