#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
set -euo pipefail

# Evaluate every trained Marformer under a run's results tree.
#
# run_marformer_eval.sh globs only RESULTS/.../<AXIS>/*_MARFORMER, one level
# deep, so it cannot see the per-seed subtrees submit_all.sh --seeds creates
# (<AXIS>/seed<N>/<run>_MARFORMER). This walks the tree instead.
#
# Unlike ablation_eval.sbatch this does NOT clear TEST_RESULTS first; it writes
# a superset (best + last + every periodic checkpoint), so it can be run over a
# tree that already holds canonical results without destroying them. Set
# CLEAR=1 to force a clean re-evaluation.
#
# Required env: RUN_ID.
# Optional: CLEAR=1 to re-evaluate from scratch;
#           INCLUDE=<substring> / EXCLUDE=<substring> to restrict which run
#           directories are walked. A validation run holds both the phase-A
#           search tree and the phase-B refit tree, and only the refit models
#           are reportable -- scoring the search tree first just delays the
#           numbers that matter.

if [[ -n "${RERUN_DIR:-}" && -f "${RERUN_DIR}/config/cluster.sh" ]]; then
    source "${RERUN_DIR}/config/cluster.sh"
else
    source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/config/cluster.sh"
fi
if [[ -n "${RERUN_CONF:-}" ]]; then
    # shellcheck disable=SC1090
    source "${RERUN_CONF}"
fi

export REQUIRE_GPU=1
rerun_activate_env

# ---------------------------------------------------------------- cell arrays
# CELL_RUN_IDS switches the array axis from "size" to "run id": each task picks
# its own RUN_ID out of the list and does the single size in SIZES. The
# obs-density thread needs this because its 18 cells are 18 runs at ONE size,
# and one array of 18 schedules where 18 arrays of 1 would just queue (the
# account runs ~10 jobs at a time -- see the task-pool notes).
# EPOCH_TABLE_NAME, if given, resolves to that file inside the task's own run
# dir, because the epoch key (FAMILY|AXIS|SIZE|SEED) is identical across cells
# and a shared table would collide.
# Empty by default, so every existing submission behaves exactly as before.
if [[ -n "${CELL_RUN_IDS:-}" ]]; then
    IFS=',' read -r -a _CELL_LIST <<< "${CELL_RUN_IDS}"
    _CELL_IDX="${SLURM_ARRAY_TASK_ID:-0}"
    [[ "${_CELL_IDX}" -lt "${#_CELL_LIST[@]}" ]] || {
        echo "FATAL: array task ${_CELL_IDX} outside CELL_RUN_IDS (${#_CELL_LIST[@]} cells)" >&2
        exit 1; }
    RUN_ID="${_CELL_LIST[${_CELL_IDX}]}"
    DATA_RUN_ID="${DATA_RUN_ID:-${RUN_ID}}"
    [[ -n "${EPOCH_TABLE_NAME:-}" ]] && EPOCH_TABLE="$(rerun_run_dir "${RUN_ID}")/${EPOCH_TABLE_NAME}"
    # SIZES holds a single entry in this mode; the array index is spent on cells.
    SLURM_ARRAY_TASK_ID=0
    echo "[cell-array] task ${_CELL_IDX} -> RUN_ID=${RUN_ID}"
fi

: "${RUN_ID:?RUN_ID is required}"
RESULTS_ROOT="$(rerun_results_root "${RUN_ID}")"

mapfile -t RUN_DIRS < <(find "${RESULTS_ROOT}" -maxdepth 6 -type d -name '*_MARFORMER' | sort \
    | { [[ -n "${INCLUDE:-}" ]] && grep -F -- "${INCLUDE}" || cat; } \
    | { [[ -n "${EXCLUDE:-}" ]] && grep -vF -- "${EXCLUDE}" || cat; })
[[ ${#RUN_DIRS[@]} -gt 0 ]] || { echo "no Marformer runs under ${RESULTS_ROOT}" >&2; exit 1; }

echo "=============================================================="
echo " tree eval | RUN_ID=${RUN_ID} | ${#RUN_DIRS[@]} runs"
echo "=============================================================="
cd "${RANKING_ROOT}"
export PYTHONUNBUFFERED=1

FAILED=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo
    echo "--- ${RUN_DIR#"${RESULTS_ROOT}"/} ---"
    [[ -f "${RUN_DIR}/checkpoints/last.ckpt" ]] || {
        echo "  skipped (no last.ckpt -- training did not finish)"
        FAILED=$((FAILED + 1))
        continue
    }
    [[ "${CLEAR:-0}" == "1" ]] && rm -rf "${RUN_DIR}/TEST_RESULTS"
    # best + last for the paper's plots, then every periodic checkpoint so the
    # epoch curve is available for the validation work and the artifact.
    rerun_run python -u -m imputer.entity_mf.test --run-dir "${RUN_DIR}" --checkpoint both --device cuda
    rerun_run python -u -m imputer.entity_mf.test --run-dir "${RUN_DIR}" --checkpoint all  --device cuda
done

echo
echo "done: ${RESULTS_ROOT}  (${FAILED} run(s) skipped)"
[[ "${FAILED}" -eq 0 ]]
