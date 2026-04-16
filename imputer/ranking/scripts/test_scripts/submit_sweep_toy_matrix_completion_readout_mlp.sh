#!/usr/bin/env bash
# Submit toy matrix-completion readout-MLP sweep via sbatch_adapt.
# Uses standalone per-config run scripts (no config env flags required).
#
# Usage (from imputer/ranking):
#   ./scripts/test_scripts/submit_sweep_toy_matrix_completion_readout_mlp.sh
#
# Preview only:
#   PREVIEW=1 ./scripts/test_scripts/submit_sweep_toy_matrix_completion_readout_mlp.sh
#
# Common resource overrides:
#   TIME=24:00:00 GPUS=1 PARTITION=gpu ./scripts/test_scripts/submit_sweep_toy_matrix_completion_readout_mlp.sh

set -euo pipefail

SBATCH_ADAPT="${SBATCH_ADAPT:-${HOME}/bin/sbatch_adapt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PARTITION="${PARTITION:-cpu}"
export GPUS="${GPUS:-0}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export TIME="${TIME:-8:00:00}"
submit_one() {
  local job_name=$1
  local script_rel=$2
  if [[ "${PREVIEW:-0}" == 1 ]]; then
    echo JOB_NAME="${job_name}" PARTITION="${PARTITION}" GPUS="${GPUS}" \
      CPUS_PER_TASK="${CPUS_PER_TASK}" TIME="${TIME}" \
      "${SBATCH_ADAPT}" "${script_rel}"
    return 0
  fi

  JOB_NAME="${job_name}" PARTITION="${PARTITION}" GPUS="${GPUS}" \
    CPUS_PER_TASK="${CPUS_PER_TASK}" TIME="${TIME}" \
    "${SBATCH_ADAPT}" "${script_rel}"
}

submit_one mc_ro_baseline scripts/test_scripts/readout_mlp_ablation/run_mc_readout_baseline.sh
submit_one mc_ro_l1_d64 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L1_D64.sh
submit_one mc_ro_l1_d128 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L1_D128.sh
submit_one mc_ro_l1_d256 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L1_D256.sh
submit_one mc_ro_l2_d64 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L2_D64.sh
submit_one mc_ro_l2_d128 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L2_D128.sh
submit_one mc_ro_l2_d256 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L2_D256.sh
submit_one mc_ro_l3_d64 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L3_D64.sh
submit_one mc_ro_l3_d128 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L3_D128.sh
submit_one mc_ro_l3_d256 scripts/test_scripts/readout_mlp_ablation/run_mc_readout_L3_D256.sh

echo "Submitted (or printed) standalone readout-MLP jobs under OUTPUT/mc_readout_mlp/"

