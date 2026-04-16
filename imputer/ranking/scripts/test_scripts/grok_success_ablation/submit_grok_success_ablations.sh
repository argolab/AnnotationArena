#!/usr/bin/env bash
# Submit all grok-success ablation jobs via sbatch_adapt (same pattern as
# scripts/test_scripts/submit_toy_matrix_completion_pointer_slurm.sh).
#
# Usage (from imputer/ranking):
#   ./scripts/test_scripts/grok_success_ablation/submit_grok_success_ablations.sh
# Preview:
#   PREVIEW=1 ./scripts/test_scripts/grok_success_ablation/submit_grok_success_ablations.sh
#
# Env: SBATCH_ADAPT (default ~/bin/sbatch_adapt), PARTITION, GPUS, CPUS_PER_TASK,
#      TIME, JOB_NAME per job. Defaults: CPU partition, no GPUs, 16 CPUs per job.

set -euo pipefail

SBATCH_ADAPT="${SBATCH_ADAPT:-${HOME}/bin/sbatch_adapt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export PARTITION="${PARTITION:-cpu}"
export GPUS="${GPUS:-0}"
export CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
export TIME="${TIME:-48:00:00}"

GROK_DIR="scripts/test_scripts/grok_success_ablation"
EXTRA_SBATCH=(--export=ALL)
# Optional Slurm name prefix when running multiple puzzle batches (e.g. g4x4r1_).
GROK_JOB_PREFIX="${GROK_JOB_PREFIX:-}"

submit() {
  local job_name=$1
  local script_rel=$2
  local full_name="${GROK_JOB_PREFIX}${job_name}"
  if [[ "${PREVIEW:-0}" == 1 ]]; then
    echo JOB_NAME="${full_name}" PARTITION="${PARTITION}" GPUS="${GPUS}" \
      CPUS_PER_TASK="${CPUS_PER_TASK}" TIME="${TIME}" \
      "${SBATCH_ADAPT}" "${script_rel}" "${EXTRA_SBATCH[@]}"
    return 0
  fi
  JOB_NAME="${full_name}" PARTITION="${PARTITION}" GPUS="${GPUS}" \
    CPUS_PER_TASK="${CPUS_PER_TASK}" TIME="${TIME}" \
    "${SBATCH_ADAPT}" "${script_rel}" "${EXTRA_SBATCH[@]}"
}

submit grok_baseline                  "${GROK_DIR}/run_grok_baseline.sh"
submit grok_no_show_correct           "${GROK_DIR}/ablation_no_show_correct_vector.sh"
submit grok_no_mult_head              "${GROK_DIR}/ablation_no_multiplication_head.sh"
submit grok_no_mult_no_oracle         "${GROK_DIR}/ablation_no_mult_no_show_correct_vector.sh"
# submit grok_layers_m1                 "${GROK_DIR}/ablation_layers_minus1.sh"
# submit grok_layers_m2                 "${GROK_DIR}/ablation_layers_minus2.sh"
# submit grok_heads_m1                  "${GROK_DIR}/ablation_heads_minus1.sh"
# submit grok_heads_m2                  "${GROK_DIR}/ablation_heads_minus2.sh"
# submit grok_emb_24                    "${GROK_DIR}/ablation_emb_24.sh"
# submit grok_emb_16                    "${GROK_DIR}/ablation_emb_16.sh"
# submit grok_dff_96                    "${GROK_DIR}/ablation_d_ff_96.sh"
# submit grok_dff_64                    "${GROK_DIR}/ablation_d_ff_64.sh"
# submit grok_lr_1e4                    "${GROK_DIR}/ablation_lr_1e-4.sh"
# submit grok_lr_1e3                    "${GROK_DIR}/ablation_lr_1e-3.sh"
# submit grok_dropout_toy               "${GROK_DIR}/ablation_dropout_toy_default.sh"

_out_root="${GROK_OUT_ROOT:-OUTPUT/grok_success_ablation}"
echo "Submitted (or printed) grok ablation jobs under ${_out_root}/ (per script OUT_DIR)."
