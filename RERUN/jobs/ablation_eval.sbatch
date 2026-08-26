#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --comment=accept_cost
#
# Evaluate the residual-stream ablation runs.
#
# run_marformer_eval.sh globs only RESULTS/.../ITEM/*_MARFORMER, one level deep,
# so it cannot see the ablation tree, which is
# RESULTS/MARFORMER/DOMAIN3-SIGMOID/ITEM/<variant>/seed<N>/<run>_MARFORMER.
# This walks the tree instead.
#
# Both checkpoint sets are written:
#   --checkpoint all   every periodic checkpoint scored on test
#   --checkpoint last  the final epoch, no selection at all
# The ablation is compared on `last`. Domain 3 has no validation split (see
# reports/OVERFITTING.md), so a test-argmin would let each arm pick its own
# luckiest checkpoint and the comparison would mostly measure selection noise.
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
RESULTS_ROOT="$(rerun_results_root "${RUN_ID}")"

mapfile -t RUN_DIRS < <(find "${RESULTS_ROOT}" -maxdepth 6 -type d -name '*_MARFORMER' | sort)
[[ ${#RUN_DIRS[@]} -gt 0 ]] || { echo "no runs under ${RESULTS_ROOT}" >&2; exit 1; }

echo "evaluating ${#RUN_DIRS[@]} ablation runs under ${RESULTS_ROOT}"
cd "${RANKING_ROOT}"
export PYTHONUNBUFFERED=1

for RUN_DIR in "${RUN_DIRS[@]}"; do
    echo
    echo "--- ${RUN_DIR#${RESULTS_ROOT}/} ---"
    rm -rf "${RUN_DIR}/TEST_RESULTS"
    rerun_run python -u -m imputer.entity_mf.test --run-dir "${RUN_DIR}" --checkpoint all  --device cuda
    rerun_run python -u -m imputer.entity_mf.test --run-dir "${RUN_DIR}" --checkpoint last --device cuda
done

echo
echo "done: ${RESULTS_ROOT}"
