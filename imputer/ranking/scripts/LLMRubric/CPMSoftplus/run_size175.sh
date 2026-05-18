#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=CPMSP_LLMR_175
#SBATCH --output=/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --mail-user=psingh54@jhu.edu

set -eo pipefail

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
export TMPDIR="${TMPDIR:-/tmp/${USER:-user}/llmrubric_cpm_softplus/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${REPO_ROOT}/logs"

SCRIPT_START=$SECONDS

SPLIT="LLMRubric_225_25_9_175"
DATA_DIR="DATA/LLM_RUBRIC/${SPLIT}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
CONFIGS_PATH="${DATA_DIR}/configs.json"
OUTPUT_DIR="RESULTS/STAN/LLM_RUBRIC/CPMSoftplus"
RUN_NAME="${SPLIT}_CPMSoftplus"
STAN_FILE="STAN/stan_models/tensor_model_softplus.stan"
CSV_PATTERN="tensor_model_softplus-*.csv"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42
ALPHA_LLM=5.0

STAGED_STAN_DIR="${XDG_CACHE_HOME}/llmrubric_cpm_softplus_stan/${SLURM_JOB_ID:-$$}"
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

TMP_CONFIG="$(mktemp "${TMPDIR}/llmrubric_cpm_softplus_config.XXXXXX.json")"
trap 'rm -f "${TMP_CONFIG}"' EXIT
write_temp_tensor_config "${CONFIGS_PATH}" "${TMP_CONFIG}"

echo ""
echo "============================================================"
echo " LLMRubric | CPM-Softplus | ${SPLIT}"
echo "============================================================"
echo " REPO_ROOT         : ${REPO_ROOT}"
echo " DATA_DIR          : ${DATA_DIR}"
echo " DATA_BUNDLE       : ${DATA_BUNDLE}"
echo " CONFIGS_PATH      : ${CONFIGS_PATH}"
echo " TMP_CONFIG        : ${TMP_CONFIG}"
echo " OUTPUT_DIR        : ${OUTPUT_DIR}"
echo " RUN_NAME          : ${RUN_NAME}"
echo " STAN_FILE         : ${STAN_FILE}"
echo " LOCAL_STAN_FILE   : ${LOCAL_STAN_FILE}"
echo " CHAINS            : ${CHAINS}"
echo " ITER_WARMUP       : ${ITER_WARMUP}"
echo " ITER_SAMPLING     : ${ITER_SAMPLING}"
echo " ADAPT_DELTA       : ${ADAPT_DELTA}"
echo " MAX_TREEDEPTH     : ${MAX_TREEDEPTH}"
echo " SEED              : ${SEED}"
echo " ALPHA_LLM         : ${ALPHA_LLM}"
echo "============================================================"

echo "[1/2] Running CPM-Softplus inference..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle        "${DATA_BUNDLE}" \
    --configs            "${TMP_CONFIG}" \
    --output-dir         "${OUTPUT_DIR}" \
    --run-name           "${RUN_NAME}" \
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
    --data-bundle        "${DATA_BUNDLE}" \
    --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir         "${OUTPUT_DIR}" \
    --run-name           "${RUN_NAME}_eval" \
    --csv-pattern        "${CSV_PATTERN}" \
    --overwrite-existing-data \
    --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))

echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
