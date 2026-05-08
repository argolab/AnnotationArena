#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

DOMAIN3_DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DOMAIN3_DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"

if [[ -n "${DOMAIN3_REPO_ROOT_OVERRIDE:-}" && -f "${DOMAIN3_REPO_ROOT_OVERRIDE}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_REPO_ROOT_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${DOMAIN3_DEFAULT_CLUSTER_ROOT}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_CLUSTER_ROOT}"
elif [[ -f "${DOMAIN3_DEFAULT_LOCAL_ROOT}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_LOCAL_ROOT}"
else
    DOMAIN3_REPO_ROOT="$(pwd)"
fi

export DOMAIN3_REPO_ROOT_OVERRIDE="${DOMAIN3_REPO_ROOT}"
# shellcheck disable=SC1091
source "${DOMAIN3_REPO_ROOT}/scripts/DOMAIN3/common.sh"

FAMILY_LABEL=""
RESULTS_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family-label)
            FAMILY_LABEL="$2"
            shift 2
            ;;
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${FAMILY_LABEL}" || -z "${RESULTS_ROOT}" ]]; then
    echo "Missing required arguments to run_marformer_eval.sh" >&2
    exit 1
fi

domain3_setup_env
EXP_NAME="domain3_final_${FAMILY_LABEL,,}_eval"
EXP_NAME="${EXP_NAME// /_}"
domain3_enable_file_logging "${EXP_NAME}"
domain3_enter_repo

SCRIPT_START=$SECONDS
DEVICE="$(domain3_choose_torch_device)"

domain3_print_block "DOMAIN3-FINAL | ${FAMILY_LABEL} | checkpoint evaluation"
echo " RESULTS_ROOT : ${RESULTS_ROOT}"
echo " REPO_ROOT    : ${DOMAIN3_REPO_ROOT}"
echo " LOG_FILE     : ${DOMAIN3_LOG_FILE}"
echo " DEVICE       : ${DEVICE}"

shopt -s nullglob
RUN_DIRS=( "${RESULTS_ROOT}/ITEM"/*_MARFORMER "${RESULTS_ROOT}/ANNOT"/*_MARFORMER )
shopt -u nullglob

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "No Marformer run directories found under ${RESULTS_ROOT}/ITEM or ${RESULTS_ROOT}/ANNOT" >&2
    exit 1
fi

for RUN_DIR in "${RUN_DIRS[@]}"; do
    if [[ ! -d "${RUN_DIR}" ]]; then
        continue
    fi

    RUN_NAME="$(basename "${RUN_DIR}")"
    RUN_START=$SECONDS
    TEST_RESULTS_DIR="${RUN_DIR}/TEST_RESULTS"

    echo ""
    echo "------------------------------------------------------------"
    echo " RUN_NAME : ${RUN_NAME}"
    echo " RUN_DIR  : ${RUN_DIR}"
    echo "------------------------------------------------------------"

    rm -rf "${TEST_RESULTS_DIR}"

    python -u -m imputer.entity_mf.test \
        --run-dir "${RUN_DIR}" \
        --checkpoint all \
        --device "${DEVICE}"

    python -u -m imputer.entity_mf.test \
        --run-dir "${RUN_DIR}" \
        --checkpoint last \
        --device "${DEVICE}"

    python - "${RUN_DIR}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
test_dir = run_dir / "TEST_RESULTS"

candidates = []
for path in sorted(test_dir.glob("*.json")):
    if path.name == "best.json":
        continue
    with open(path) as f:
        data = json.load(f)
    missing = data.get("missing", {})
    log_loss = missing.get("log_loss")
    rmse = missing.get("rmse")
    n = missing.get("n", 0)
    if log_loss is None or n == 0:
        continue
    candidates.append((float(log_loss), float(rmse) if rmse is not None else math.inf, path.name, data))

if not candidates:
    raise SystemExit(f"No valid checkpoint results found in {test_dir}")

candidates.sort(key=lambda x: (x[0], x[1], x[2]))
best_log_loss, best_rmse, best_name, best_data = candidates[0]
best_data["selected_from"] = best_name
best_data["selection_metric"] = "missing.log_loss"

best_path = test_dir / "best.json"
with open(best_path, "w") as f:
    json.dump(best_data, f, indent=2)

print(f"  best checkpoint: {best_name}")
print(f"  best missing log_loss={best_log_loss:.6f} rmse={best_rmse:.6f}")
print(f"  saved -> {best_path}")
PY

    RUN_ELAPSED=$(( SECONDS - RUN_START ))
    PAYLOAD_JSON=$(cat <<EOF
{
  "launcher_type": "marformer_eval",
  "family": "${FAMILY_LABEL}",
  "run_name": "${RUN_NAME}",
  "run_dir": "${RUN_DIR}",
  "test_results_dir": "${TEST_RESULTS_DIR}",
  "elapsed_seconds": ${RUN_ELAPSED},
  "device": "${DEVICE}",
  "selection_metric": "missing.log_loss"
}
EOF
)
    domain3_write_json_payload "${TEST_RESULTS_DIR}/launcher_metadata.json" "${PAYLOAD_JSON}"

    echo "  ↳ completed in $(( RUN_ELAPSED / 60 ))m $(( RUN_ELAPSED % 60 ))s"
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
domain3_print_block "DOMAIN3-FINAL | ${FAMILY_LABEL} | checkpoint evaluation complete"
echo " TOTAL_TIME  : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " RESULTS_ROOT: ${RESULTS_ROOT}"
