#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

FAMILY_LABEL=""
MODEL_LABEL=""
MODEL_TAG=""
STAN_FILE=""
OUTPUT_ROOT=""
AXIS=""
SIZES_CSV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family-label)
            FAMILY_LABEL="$2"
            shift 2
            ;;
        --model-label)
            MODEL_LABEL="$2"
            shift 2
            ;;
        --model-tag)
            MODEL_TAG="$2"
            shift 2
            ;;
        --stan-file)
            STAN_FILE="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --axis)
            AXIS="$2"
            shift 2
            ;;
        --sizes)
            SIZES_CSV="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${FAMILY_LABEL}" || -z "${MODEL_LABEL}" || -z "${MODEL_TAG}" || -z "${STAN_FILE}" || -z "${OUTPUT_ROOT}" || -z "${AXIS}" || -z "${SIZES_CSV}" ]]; then
    echo "Missing required arguments to run_stan_group.sh" >&2
    exit 1
fi

domain3_setup_env
domain3_enter_repo

SCRIPT_START=$SECONDS

DATA_ROOT="DATA/STAN/DOMAIN3-FINAL"
CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

LOCAL_STAN_FILE="$(domain3_stage_stan_file "${STAN_FILE}")"
CSV_PATTERN="$(basename "${LOCAL_STAN_FILE%.stan}")-*.csv"

IFS=',' read -r -a SIZE_LIST <<< "${SIZES_CSV}"

domain3_print_block "DOMAIN3-FINAL | ${FAMILY_LABEL} | ${MODEL_LABEL} | ${AXIS}"
echo " OUTPUT_ROOT         : ${OUTPUT_ROOT}"
echo " STAN_FILE           : ${STAN_FILE}"
echo " STAGED_STAN_FILE    : ${LOCAL_STAN_FILE}"
echo " CSV_PATTERN         : ${CSV_PATTERN}"
echo " SIZES               : ${SIZE_LIST[*]}"
echo " CHAINS              : ${CHAINS}"
echo " ITER_WARMUP         : ${ITER_WARMUP}"
echo " ITER_SAMPLING       : ${ITER_SAMPLING}"
echo " ADAPT_DELTA         : ${ADAPT_DELTA}"
echo " MAX_TREEDEPTH       : ${MAX_TREEDEPTH}"
echo " SEED                : ${SEED}"

for RAW_SIZE in "${SIZE_LIST[@]}"; do
    SIZE="$(echo "${RAW_SIZE}" | xargs)"
    RUN_START=$SECONDS

    if [[ "${AXIS}" == "item" ]]; then
        SPLIT_NAME="DOMAIN3-FINAL_Item_T_${SIZE}"
        DATA_DIR="${DATA_ROOT}/ItemSplits/Transductive/${SPLIT_NAME}"
    else
        SPLIT_NAME="DOMAIN3-FINAL_Annot_T_${SIZE}"
        DATA_DIR="${DATA_ROOT}/AnnotSplits/Transductive/${SPLIT_NAME}"
    fi

    DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
    RUN_NAME="${SPLIT_NAME}_${MODEL_TAG}"
    FIT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    EVAL_DIR="${OUTPUT_ROOT}/${RUN_NAME}_eval"

    echo ""
    echo "------------------------------------------------------------"
    echo " RUN_NAME  : ${RUN_NAME}"
    echo " DATA_DIR   : ${DATA_DIR}"
    echo " FIT_DIR    : ${FIT_DIR}"
    echo " EVAL_DIR   : ${EVAL_DIR}"
    echo "------------------------------------------------------------"

    FIT_START=$SECONDS
    echo "[1/2] Running MCMC..."
    python STAN/stan_code/scripts/run_inference.py \
        --data-bundle        "${DATA_BUNDLE}"           \
        --configs            "${DATA_DIR}/configs.json" \
        --output-dir         "${OUTPUT_ROOT}"           \
        --run-name           "${RUN_NAME}"              \
        --stan-type          "tensor"                   \
        --stan-file          "${LOCAL_STAN_FILE}"       \
        --chains             "${CHAINS}"                \
        --iter-warmup        "${ITER_WARMUP}"           \
        --iter-sampling      "${ITER_SAMPLING}"         \
        --adapt-delta        "${ADAPT_DELTA}"           \
        --max-treedepth      "${MAX_TREEDEPTH}"         \
        --seed               "${SEED}"                  \
        --overwrite-existing-data
    FIT_ELAPSED=$(( SECONDS - FIT_START ))

    FIT_PAYLOAD_JSON=$(cat <<EOF
{
  "launcher_type": "stan_fit",
  "family": "${FAMILY_LABEL}",
  "model_label": "${MODEL_LABEL}",
  "model_tag": "${MODEL_TAG}",
  "axis": "${AXIS}",
  "protocol": "transductive",
  "size": ${SIZE},
  "data_dir": "${DATA_DIR}",
  "run_name": "${RUN_NAME}",
  "output_dir": "${FIT_DIR}",
  "stan_file": "${STAN_FILE}",
  "staged_stan_file": "${LOCAL_STAN_FILE}",
  "csv_pattern": "${CSV_PATTERN}",
  "fit_elapsed_seconds": ${FIT_ELAPSED},
  "mcmc": {
    "chains": ${CHAINS},
    "iter_warmup": ${ITER_WARMUP},
    "iter_sampling": ${ITER_SAMPLING},
    "adapt_delta": ${ADAPT_DELTA},
    "max_treedepth": ${MAX_TREEDEPTH},
    "seed": ${SEED}
  }
}
EOF
)
    domain3_write_json_payload "${FIT_DIR}/launcher_metadata.json" "${FIT_PAYLOAD_JSON}"

    EVAL_START=$SECONDS
    echo "[2/2] Evaluating posterior predictions..."
    python STAN/stan_code/scripts/evaluate_predictions.py \
        --data-bundle        "${DATA_BUNDLE}" \
        --mcmc-dir           "${FIT_DIR}"     \
        --output-dir         "${OUTPUT_ROOT}" \
        --run-name           "${RUN_NAME}_eval" \
        --csv-pattern        "${CSV_PATTERN}" \
        --overwrite-existing-data             \
        --verbose
    EVAL_ELAPSED=$(( SECONDS - EVAL_START ))

    RUN_ELAPSED=$(( SECONDS - RUN_START ))
    EVAL_PAYLOAD_JSON=$(cat <<EOF
{
  "launcher_type": "stan_eval",
  "family": "${FAMILY_LABEL}",
  "model_label": "${MODEL_LABEL}",
  "model_tag": "${MODEL_TAG}",
  "axis": "${AXIS}",
  "protocol": "transductive",
  "size": ${SIZE},
  "data_dir": "${DATA_DIR}",
  "fit_dir": "${FIT_DIR}",
  "eval_dir": "${EVAL_DIR}",
  "stan_file": "${STAN_FILE}",
  "staged_stan_file": "${LOCAL_STAN_FILE}",
  "csv_pattern": "${CSV_PATTERN}",
  "fit_elapsed_seconds": ${FIT_ELAPSED},
  "eval_elapsed_seconds": ${EVAL_ELAPSED},
  "total_elapsed_seconds": ${RUN_ELAPSED},
  "mcmc": {
    "chains": ${CHAINS},
    "iter_warmup": ${ITER_WARMUP},
    "iter_sampling": ${ITER_SAMPLING},
    "adapt_delta": ${ADAPT_DELTA},
    "max_treedepth": ${MAX_TREEDEPTH},
    "seed": ${SEED}
  }
}
EOF
)
    domain3_write_json_payload "${EVAL_DIR}/launcher_metadata.json" "${EVAL_PAYLOAD_JSON}"

    echo "  ↳ fit:  $(( FIT_ELAPSED / 60 ))m $(( FIT_ELAPSED % 60 ))s"
    echo "  ↳ eval: $(( EVAL_ELAPSED / 60 ))m $(( EVAL_ELAPSED % 60 ))s"
    echo "  ↳ total: $(( RUN_ELAPSED / 60 ))m $(( RUN_ELAPSED % 60 ))s"
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
domain3_print_block "DOMAIN3-FINAL | ${FAMILY_LABEL} | ${MODEL_LABEL} | ${AXIS} complete"
echo " TOTAL_TIME  : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"

