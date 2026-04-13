#!/usr/bin/env bash
# Submit 18 Slurm jobs (layers × heads × lr), each running the full 4-way ablation on 2×2, D=1.
# Uses sbatch_adapt the same way as scripts/test_scripts/submit_toy_matrix_completion_pointer_slurm.sh:
#   - Optional env before the command: JOB_NAME, GPUS, TIME, PARTITION, ...
#   - Path is relative to imputer/ranking after cd.
#   - Passes hyperparameters into the worker via exported env; worker must use --export=ALL
#     so LAYERS, HEADS, LR, EMB, OUT_ROOT reach the batch script.
#
# Usage (from anywhere):
#   ./scripts/test_scripts/submit_sweep_toy_matrix_completion_2x2_ablation.sh
# Preview commands only:
#   PREVIEW=1 ./scripts/test_scripts/submit_sweep_toy_matrix_completion_2x2_ablation.sh
# Override adapter or resources:
#   SBATCH_ADAPT=~/bin/sbatch_adapt GPUS=1 TIME=8:00:00 OUT_ROOT=OUTPUT/my_sweep ./scripts/test_scripts/...
#
# Sweep: layers {2,4,6} × heads {2,4} × lr {1e-4, 1e-3}; EMB=16, STEPS=2000 by default.

set -euo pipefail

SBATCH_ADAPT="${SBATCH_ADAPT:-${HOME}/bin/sbatch_adapt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export GPUS="${GPUS:-1}"
export TIME="${TIME:-8:00:00}"
export STEPS="${STEPS:-2000}"
export EMB="${EMB:-16}"
export D_FF="${D_FF:-64}"
export SEED="${SEED:-0}"
export OUT_ROOT="${OUT_ROOT:-OUTPUT/toy_matrix_completion_2x2_sweep}"

WORKER="scripts/test_scripts/run_toy_matrix_completion_2x2_ablation.sh"
EXTRA_SBATCH=(--export=ALL)

submit_combo() {
  local L=$1 H=$2 LR=$3
  # Slurm job name: short, no odd characters
  local jname
  jname="tmc_L${L}H${H}_$(printf '%s' "$LR" | sed 's/\./p/g; s/-/_/g')"

  if [[ "${PREVIEW:-0}" == 1 ]]; then
    echo JOB_NAME="${jname}" GPUS="${GPUS}" TIME="${TIME}" \
      LAYERS="${L}" HEADS="${H}" LR="${LR}" EMB="${EMB}" D_FF="${D_FF}" STEPS="${STEPS}" SEED="${SEED}" OUT_ROOT="${OUT_ROOT}" \
      "${SBATCH_ADAPT}" "${WORKER}" "${EXTRA_SBATCH[@]}"
    return 0
  fi
  JOB_NAME="${jname}" GPUS="${GPUS}" TIME="${TIME}" \
    LAYERS="${L}" HEADS="${H}" LR="${LR}" EMB="${EMB}" D_FF="${D_FF}" STEPS="${STEPS}" SEED="${SEED}" OUT_ROOT="${OUT_ROOT}" \
    "${SBATCH_ADAPT}" "${WORKER}" "${EXTRA_SBATCH[@]}"
}

# --- layers=2 ---
submit_combo 2 2 1e-4
submit_combo 2 2 1e-3
submit_combo 2 4 1e-4
submit_combo 2 4 1e-3
# --- layers=4 ---
submit_combo 4 2 1e-4
submit_combo 4 2 1e-3
submit_combo 4 4 1e-4
submit_combo 4 4 1e-3
# --- layers=6 ---
submit_combo 6 2 1e-4
submit_combo 6 2 1e-3
submit_combo 6 4 1e-4
submit_combo 6 4 1e-3

echo "Submitted (or printed) ${OUT_ROOT}/L<layers>_H<heads>_Emb<emb>_lr<lr>/{oracle_on_mult_on,...}"
