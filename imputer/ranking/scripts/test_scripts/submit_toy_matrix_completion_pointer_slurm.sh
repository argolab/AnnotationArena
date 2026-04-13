#!/usr/bin/env bash
# Submit the four toy matrix-completion jobs (1 GPU, 48h each) via sbatch_adapt.
# Run from any directory; changes to imputer/ranking before sbatch.
#
# Usage:
#   ./submit_toy_matrix_completion_pointer_slurm.sh
# Writes under OUTPUT/toy_matrix_completion_ablation_fresh/{small_no_pointer,...} by default.
# Optional:
#   STEPS=50000 ./submit_toy_matrix_completion_pointer_slurm.sh
#   TOY_MC_OUT_ROOT=OUTPUT/my_new_batch ./submit_toy_matrix_completion_pointer_slurm.sh
#   PREVIEW=1 ./submit_toy_matrix_completion_pointer_slurm.sh     # print sbatch_adapt lines only

set -euo pipefail

SBATCH_ADAPT="${SBATCH_ADAPT:-$HOME/bin/sbatch_adapt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GPUS="${GPUS:-1}"
export TIME="${TIME:-48:00:00}"

EXTRA_SBATCH=()
NEED_EXPORT=false
if [[ -n "${STEPS:-}" ]]; then
  export STEPS
  NEED_EXPORT=true
fi
if [[ -n "${TOY_MC_OUT_ROOT:-}" ]]; then
  export TOY_MC_OUT_ROOT
  NEED_EXPORT=true
fi
if [[ "$NEED_EXPORT" == true ]]; then
  EXTRA_SBATCH+=(--export=ALL)
fi

submit() {
  local job_name="$1"
  local script_rel="$2"
  if [[ "${PREVIEW:-0}" == 1 ]]; then
    echo JOB_NAME="${job_name}" GPUS="${GPUS}" TIME="${TIME}" "${SBATCH_ADAPT}" "${script_rel}" "${EXTRA_SBATCH[@]}"
    return 0
  fi
  JOB_NAME="${job_name}" GPUS="${GPUS}" TIME="${TIME}" \
    "${SBATCH_ADAPT}" "${script_rel}" "${EXTRA_SBATCH[@]}"
}

submit toy_mc_small_noptr scripts/test_scripts/run_toy_matrix_completion_small_noptr.sh
submit toy_mc_small_ptr scripts/test_scripts/run_toy_matrix_completion_small_pointer.sh
submit toy_mc_big_noptr scripts/test_scripts/run_toy_matrix_completion_big_noptr.sh
submit toy_mc_big_ptr scripts/test_scripts/run_toy_matrix_completion_big_pointer.sh
