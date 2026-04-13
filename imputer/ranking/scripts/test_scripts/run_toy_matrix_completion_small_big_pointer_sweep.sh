#!/usr/bin/env bash
# Run all four toy matrix-completion variants in sequence (same as the four standalone scripts).
# Results: OUTPUT/toy_matrix_completion_ablation_fresh/{small_no_pointer,small_with_pointer,big_no_pointer,big_with_pointer}
# Optional: TOY_MC_OUT_ROOT=OUTPUT/other_batch STEPS=50000 ./run_toy_matrix_completion_small_big_pointer_sweep.sh
# Subset:   RUNS=small_noptr,big_noptr ./run_toy_matrix_completion_small_big_pointer_sweep.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS="${RUNS:-small_noptr,small_ptr,big_noptr,big_ptr}"
IFS=',' read -r -a RUN_LIST <<< "${RUNS}"

for tag in "${RUN_LIST[@]}"; do
  tag="$(echo "${tag}" | xargs)"
  case "${tag}" in
    small_noptr) bash "${DIR}/run_toy_matrix_completion_small_noptr.sh" ;;
    small_ptr) bash "${DIR}/run_toy_matrix_completion_small_pointer.sh" ;;
    big_noptr) bash "${DIR}/run_toy_matrix_completion_big_noptr.sh" ;;
    big_ptr) bash "${DIR}/run_toy_matrix_completion_big_pointer.sh" ;;
    *)
      echo "Unknown RUNS tag: ${tag} (use small_noptr, small_ptr, big_noptr, big_ptr)" >&2
      exit 1
      ;;
  esac
done

echo "All requested runs finished."
