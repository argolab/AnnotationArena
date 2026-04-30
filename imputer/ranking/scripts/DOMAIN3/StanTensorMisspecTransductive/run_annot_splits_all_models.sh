#!/bin/bash

# Run all DOMAIN3 annotator-split transductive tensor misspecified models.
set -euo pipefail

export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

SCRIPT_START=$SECONDS

OUTPUT_DIR="RESULTS/STAN/TENSOR/DOMAIN3_MISSPEC/ANNOT"
SIZE_LIST=(5 10 15 20 25)

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

MODELS=(
  "MISP_PROJ:tensor_shared_annotator_projection.stan"
  "MISP_BIN_RAWZ:tensor_shared_annotator_binning_raw_z.stan"
  "MISP_PROJ_BIN_RAWZ:tensor_shared_annotator_projection_binning_raw_z.stan"
)

echo "============================================================"
echo " DOMAIN3 transductive annot splits | tensor misspecified batch"
echo " Sizes       : ${SIZE_LIST[*]}"
echo " Output root : ${OUTPUT_DIR}"
echo "============================================================"

for SIZE in "${SIZE_LIST[@]}"; do
  DATA_DIR="DATA/STAN/DOMAIN3/AnnotSplits/Transductive/Tensor_400_25_9_DOMAIN3_Annot_T_${SIZE}"
  DATA_BUNDLE="${DATA_DIR}/data_bundle.json"

  for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL_TAG="${MODEL_ENTRY%%:*}"
    MODEL_FILE="${MODEL_ENTRY##*:}"
    MODEL_STEM="${MODEL_FILE%.stan}"
    RUN_NAME="Tensor_400_25_9_DOMAIN3_Annot_T_${SIZE}_${MODEL_TAG}"

    echo ""
    echo "------------------------------------------------------------"
    echo " Size      : ${SIZE}"
    echo " Model     : ${MODEL_FILE}"
    echo " Run name  : ${RUN_NAME}"
    echo "------------------------------------------------------------"

    python STAN/stan_code/scripts/run_inference.py \
      --data-bundle "${DATA_BUNDLE}" \
      --configs "${DATA_DIR}/configs.json" \
      --output-dir "${OUTPUT_DIR}" \
      --run-name "${RUN_NAME}" \
      --stan-type "tensor" \
      --stan-file "STAN/stan_models/${MODEL_FILE}" \
      --chains "${CHAINS}" \
      --iter-warmup "${ITER_WARMUP}" \
      --iter-sampling "${ITER_SAMPLING}" \
      --adapt-delta "${ADAPT_DELTA}" \
      --max-treedepth "${MAX_TREEDEPTH}" \
      --seed "${SEED}" \
      --overwrite-existing-data

    python STAN/stan_code/scripts/evaluate_predictions.py \
      --data-bundle "${DATA_BUNDLE}" \
      --mcmc-dir "${OUTPUT_DIR}/${RUN_NAME}" \
      --output-dir "${OUTPUT_DIR}" \
      --run-name "${RUN_NAME}_eval" \
      --csv-pattern "${MODEL_STEM}-*.csv" \
      --overwrite-existing-data \
      --verbose
  done
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results root: ${OUTPUT_DIR}"
echo "============================================================"
