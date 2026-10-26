#!/bin/bash
#SBATCH --job-name=CORNER
#SBATCH --partition=med
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --comment=accept_cost
# Score the refit tree on the TEST CORNER.
#
# tree_eval.sbatch runs imputer.entity_mf.test, whose "missing" metric pools the
# held-out corner with the training partition's own missing cells (D22) -- 1028
# cells against the corner's 414 at Item_T_300, and only 2.9% corner at K=4000.
# Stan is scored on the test block alone, so building a table from those two
# files directly reproduces D23: one figure, two different populations. This
# produces the corner-only number that matches Stan.
#
# --run-dir takes the RUN directory, not a family tree: the scorer appends
# RESULTS/<family>/DOMAIN3-SIGMOID itself and loops over both families. Handing
# it the resolved tree makes both families miss, and it then writes rows:[] and
# exits 0 -- so the emptiness check below is the real guard, not the exit code.
#
#   SCORE_RUN  = run id
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
# Job parameters may arrive in a conf file (submit_phase1.sh), not only as
# exported env.
if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi
# CELL_RUN_IDS makes the array axis the run id, so a multi-cell thread scores
# all its cells from one job. See the same block in jobs/marformer_train.sbatch.
if [[ -n "${CELL_RUN_IDS:-}" ]]; then
    IFS=',' read -r -a _CELL_LIST <<< "${CELL_RUN_IDS}"
    _CELL_IDX="${SLURM_ARRAY_TASK_ID:-0}"
    [[ "${_CELL_IDX}" -lt "${#_CELL_LIST[@]}" ]] || {
        echo "FATAL: array task ${_CELL_IDX} outside CELL_RUN_IDS (${#_CELL_LIST[@]} cells)" >&2
        exit 1; }
    SCORE_RUN="${_CELL_LIST[${_CELL_IDX}]}"
    echo "[cell-array] task ${_CELL_IDX} -> SCORE_RUN=${SCORE_RUN}"
fi

: "${SCORE_RUN:?}"
cd "$(dirname "${RERUN_DIR}")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
OUT="${RERUN_DIR}/runs/${SCORE_RUN}/corner_metrics"
mkdir -p "${OUT}"

python RERUN/analysis/split_missing_metrics.py \
    --run-dir "${RERUN_DIR}/runs/${SCORE_RUN}" \
    --checkpoint last --device cpu --corner-whole \
    -o "${OUT}/corner.json"

python - "${OUT}/corner.json" <<'PY'
import json, sys
from pathlib import Path
rows = json.loads(Path(sys.argv[1]).read_text()).get("rows", [])
if not rows:
    sys.exit("FATAL: the scorer produced 0 rows. It exits 0 on an empty tree, "
             "so this check is what catches a wrong --run-dir or a symlinked "
             "arm directory (rglob does not descend into symlinks).")
fams = {}
for r in rows:
    fams[(r["family"], r["axis"])] = fams.get((r["family"], r["axis"]), 0) + 1
print(f"scored {len(rows)} runs:")
for k in sorted(fams):
    print(f"  {k[0]:<12} {k[1]:<6} {fams[k]:>3} cells")
PY
echo done
