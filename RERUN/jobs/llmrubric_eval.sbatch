#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#
# Test evaluation for the LLM-Rubric Marformer runs.
#
# Two passes, deliberately:
#   --checkpoint both  writes best.json / last.json, where "best" is the
#                      checkpoint ModelCheckpoint selected on *validation*
#                      loss. plot_llmrubric_cpm_sigmoid.py reads these.
#   --checkpoint all   additionally scores every periodic checkpoint on test.
#                      Those files are what analysis/checkpoint_selection.py
#                      needs to measure how much selecting on test would have
#                      flattered the numbers (DISCREPANCIES.md D14).
# Order matters: "all" must not be allowed to overwrite best.json, so "both"
# runs last.
set -euo pipefail

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

: "${RUN_ID:?RUN_ID is required}"
SIZES="${SIZES:-10,20,30,40,50,75,100,125,150,175}"
IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
SIZE="${SIZE_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
TARGET="${RUN_DIR}/RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_${SIZE}"

[[ -d "${TARGET}/checkpoints" ]] || {
    echo "FATAL: no checkpoints in ${TARGET}" >&2
    exit 1
}

cd "${RANKING_ROOT}"
export PYTHONUNBUFFERED=1

echo "=== size=${SIZE}: scoring every checkpoint on test ==="
rerun_run python -u -m imputer.entity_mf.test --run-dir "${TARGET}" --checkpoint all --device cuda

echo "=== size=${SIZE}: writing best.json (validation-selected) and last.json ==="
rerun_run python -u -m imputer.entity_mf.test --run-dir "${TARGET}" --checkpoint both --device cuda

echo "done: ${TARGET}/TEST_RESULTS"
