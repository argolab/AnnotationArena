#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=CPMSI_LLMR_10_30
#SBATCH --output=/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --mail-user=psingh54@jhu.edu

set -eo pipefail

SIZES=(10 20 30)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"

looks_like_repo_root() {
    local candidate="$1"
    [[ -n "${candidate}" && -d "${candidate}/imputer/entity_mf" && -d "${candidate}/STAN/stan_code" ]]
}

resolve_repo_root() {
    if [[ -n "${LLMRUBRIC_REPO_ROOT_OVERRIDE:-}" ]] && looks_like_repo_root "${LLMRUBRIC_REPO_ROOT_OVERRIDE}"; then
        echo "${LLMRUBRIC_REPO_ROOT_OVERRIDE}"
        return
    fi
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && looks_like_repo_root "${SLURM_SUBMIT_DIR}"; then
        echo "${SLURM_SUBMIT_DIR}"
        return
    fi
    if looks_like_repo_root "${DEFAULT_CLUSTER_ROOT}"; then
        echo "${DEFAULT_CLUSTER_ROOT}"
        return
    fi
    if looks_like_repo_root "${DEFAULT_LOCAL_ROOT}"; then
        echo "${DEFAULT_LOCAL_ROOT}"
        return
    fi
    if looks_like_repo_root "${PWD}"; then
        echo "${PWD}"
        return
    fi
    echo "${SCRIPT_DIR}/../../.."
}

REPO_ROOT="$(resolve_repo_root)"
export LLMRUBRIC_REPO_ROOT_OVERRIDE="${REPO_ROOT}"

if [ -f "${HOME}/.bashrc" ]; then
    set +u
    # shellcheck disable=SC1090
    source "${HOME}/.bashrc"
fi
if command -v module >/dev/null 2>&1; then
    module load anaconda3/2024.02-1 >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
    set +u
    conda activate prabhav2 >/dev/null 2>&1 || true
fi
set -u

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export TMPDIR="${TMPDIR:-/tmp/${USER:-user}/llmrubric_cpm_sigmoid/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${REPO_ROOT}/logs"

SCRIPT_START=$SECONDS

OUTPUT_DIR="RESULTS/STAN/LLM_RUBRIC/CPMSigmoid"
STAN_FILE="STAN/stan_models/tensor_model_sigmoid.stan"
CSV_PATTERN="tensor_model_sigmoid-*.csv"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42
ALPHA_LLM=5.0

STAGED_STAN_DIR="${XDG_CACHE_HOME}/llmrubric_cpm_sigmoid_stan/${SLURM_JOB_ID:-$$}"
mkdir -p "${STAGED_STAN_DIR}"
LOCAL_STAN_FILE="${STAGED_STAN_DIR}/$(basename "${STAN_FILE}")"
cp "${STAN_FILE}" "${LOCAL_STAN_FILE}"

write_temp_tensor_config() {
    local src_config="$1"
    local out_config="$2"
    python - "$src_config" "$out_config" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
with open(src) as f:
    raw = json.load(f)
dg = raw.get("datagen", raw)
merged = {
    **dg,
    "D": 32,
    "T": 3,
    "alpha_confusion": 15.0,
    "use_dawid_skene_noise": False,
    "derive_thresholds_from_annotator": False,
    "stan_type": "tensor",
}
with open(dst, "w") as f:
    json.dump({"datagen": merged}, f, indent=2)
PY
}

run_one_size() {
    local size="$1"
    local split="LLMRubric_225_25_9_${size}"
    local data_dir="DATA/LLM_RUBRIC/${split}"
    local data_bundle="${data_dir}/data_bundle.json"
    local configs_path="${data_dir}/configs.json"
    local run_name="${split}_CPMSigmoid"
    local tmp_config
    tmp_config="$(mktemp "${TMPDIR}/llmrubric_cpm_sigmoid_config_${size}.XXXXXX.json")"
    write_temp_tensor_config "${configs_path}" "${tmp_config}"

    echo ""
    echo "============================================================"
    echo " LLMRubric | CPM-Sigmoid | ${split}"
    echo "============================================================"
    echo " REPO_ROOT         : ${REPO_ROOT}"
    echo " DATA_DIR          : ${data_dir}"
    echo " DATA_BUNDLE       : ${data_bundle}"
    echo " CONFIGS_PATH      : ${configs_path}"
    echo " TMP_CONFIG        : ${tmp_config}"
    echo " OUTPUT_DIR        : ${OUTPUT_DIR}"
    echo " RUN_NAME          : ${run_name}"
    echo " LOCAL_STAN_FILE   : ${LOCAL_STAN_FILE}"
    echo "============================================================"

    echo "[1/2] Running CPM-Sigmoid inference..."
    python STAN/stan_code/scripts/run_inference.py \
        --data-bundle        "${data_bundle}" \
        --configs            "${tmp_config}" \
        --output-dir         "${OUTPUT_DIR}" \
        --run-name           "${run_name}" \
        --stan-type          "tensor" \
        --stan-file          "${LOCAL_STAN_FILE}" \
        --chains             "${CHAINS}" \
        --iter-warmup        "${ITER_WARMUP}" \
        --iter-sampling      "${ITER_SAMPLING}" \
        --adapt-delta        "${ADAPT_DELTA}" \
        --max-treedepth      "${MAX_TREEDEPTH}" \
        --seed               "${SEED}" \
        --alpha-llm          "${ALPHA_LLM}" \
        --transductive-use-test-observed \
        --overwrite-existing-data

    echo "[2/2] Evaluating posterior predictions..."
    python STAN/stan_code/scripts/evaluate_predictions.py \
        --data-bundle        "${data_bundle}" \
        --mcmc-dir           "${OUTPUT_DIR}/${run_name}" \
        --output-dir         "${OUTPUT_DIR}" \
        --run-name           "${run_name}_eval" \
        --csv-pattern        "${CSV_PATTERN}" \
        --overwrite-existing-data \
        --verbose

    rm -f "${tmp_config}"
}

for size in "${SIZES[@]}"; do
    run_one_size "${size}"
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))

echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Sizes: ${SIZES[*]}"
echo " Results root: ${OUTPUT_DIR}"
echo "============================================================"
