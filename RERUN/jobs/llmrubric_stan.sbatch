#!/bin/bash
#SBATCH --job-name=LLMR-STAN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --comment=accept_cost
#
# Stan CPM-Sigmoid on the real LLM-Rubric data, one array task per training size.
#
# Replaces scripts/LLMRubric/CPMSigmoid/run_size*.sh, which hardcode Prabhav's
# /home/psingh54 output path, --partition=cpu (does not exist here), and
# `conda activate prabhav2 || true` (so it silently runs in whatever env is
# active). Those scripts also resolve every path relative to a repo root that
# must contain DATA/ and RESULTS/, which would mean symlinking the shared repo
# root while other jobs are running. Both Python entry points take explicit
# paths, so this calls them directly instead.
#
# Sampling matches the published LLM-Rubric fits exactly -- 1 chain, 300 warmup,
# 500 sampling, adapt_delta 0.85, max_treedepth 12, seed 42, alpha_llm 5.0 --
# because the point of this rerun is to check the published numbers, not to
# change the budget. See RERUN/PAPER_EDITS_SWEEP.md B12.
#
#   RUN_ID  (required)
#   SIZES   comma-separated training sizes, one per array task
set -euo pipefail
RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env
: "${RUN_ID:?RUN_ID is required}"

SIZES="${SIZES:-10,20,30,40,50,75,100,125,150,175}"
IFS=',' read -r -a SIZE_LIST <<< "${SIZES}"
SIZE="${SIZE_LIST[${SLURM_ARRAY_TASK_ID:-0}]}"

RUN_DIR="$(rerun_run_dir "${RUN_ID}")"
SPLIT="LLMRubric_225_25_9_${SIZE}"
DATA_DIR="${RUN_DIR}/DATA/LLM_RUBRIC/${SPLIT}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
CONFIGS_PATH="${DATA_DIR}/configs.json"
# plot_llmrubric_cpm_sigmoid.py reads RESULTS/STAN/CPMSigmoid while the original
# scripts wrote RESULTS/STAN/LLM_RUBRIC/CPMSigmoid (FIGURES.md, D10). Write the
# path the plot script reads; make_figures.sh's compatibility symlink then has
# nothing to bridge.
OUTPUT_DIR="${RUN_DIR}/RESULTS/STAN/CPMSigmoid"
RUN_NAME="${SPLIT}_CPMSigmoid"
STAN_FILE="${RANKING_ROOT}/STAN/stan_models/tensor_model_sigmoid.stan"
CSV_PATTERN="tensor_model_sigmoid-*.csv"

CHAINS="${LLMRUBRIC_STAN_CHAINS:-1}"
ITER_WARMUP="${LLMRUBRIC_STAN_WARMUP:-300}"
ITER_SAMPLING="${LLMRUBRIC_STAN_SAMPLING:-500}"
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42
ALPHA_LLM=5.0

for f in "${DATA_BUNDLE}" "${CONFIGS_PATH}" "${STAN_FILE}"; do
    [[ -f "${f}" ]] || { echo "FATAL: missing ${f}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"

export TMPDIR="${TMPDIR:-/tmp/${USER}/llmrubric_stan/${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"
# cmdstan compiles next to the model, so stage it somewhere writable per job
# rather than racing other array tasks over one copy in the repo.
LOCAL_STAN_FILE="${TMPDIR}/$(basename "${STAN_FILE}")"
cp "${STAN_FILE}" "${LOCAL_STAN_FILE}"

TMP_CONFIG="${TMPDIR}/config.json"
python - "${CONFIGS_PATH}" "${TMP_CONFIG}" <<'PY'
import json, sys
from pathlib import Path
raw = json.loads(Path(sys.argv[1]).read_text())
dg = raw.get("datagen", raw)
merged = {**dg, "D": 32, "T": 3, "alpha_confusion": 15.0,
          "use_dawid_skene_noise": False,
          "derive_thresholds_from_annotator": False,
          "stan_type": "tensor"}
Path(sys.argv[2]).write_text(json.dumps({"datagen": merged}, indent=2))
PY

echo "=============================================================="
echo " LLMRubric Stan CPM-Sigmoid | ${SPLIT}"
echo " DATA        : ${DATA_BUNDLE}"
echo " OUTPUT      : ${OUTPUT_DIR}/${RUN_NAME}"
echo " SAMPLING    : chains=${CHAINS} warmup=${ITER_WARMUP} sampling=${ITER_SAMPLING}"
echo "=============================================================="

cd "${RANKING_ROOT}"
export PYTHONPATH="${RANKING_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1
START=${SECONDS}

echo "[1/2] inference"
rerun_run python STAN/stan_code/scripts/run_inference.py \
    --data-bundle "${DATA_BUNDLE}" --configs "${TMP_CONFIG}" \
    --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}" \
    --stan-type tensor --stan-file "${LOCAL_STAN_FILE}" \
    --chains "${CHAINS}" --iter-warmup "${ITER_WARMUP}" \
    --iter-sampling "${ITER_SAMPLING}" --adapt-delta "${ADAPT_DELTA}" \
    --max-treedepth "${MAX_TREEDEPTH}" --seed "${SEED}" \
    --alpha-llm "${ALPHA_LLM}" \
    --transductive-use-test-observed --overwrite-existing-data

echo "[2/2] evaluation"
rerun_run python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle "${DATA_BUNDLE}" \
    --mcmc-dir "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir "${OUTPUT_DIR}" --run-name "${RUN_NAME}_eval" \
    --csv-pattern "${CSV_PATTERN}" --overwrite-existing-data --verbose

echo "done in $(( SECONDS - START ))s"
