#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

# Cluster CPU runner for evaluating the misspecified discrete Stan model on
# LLMRubric size-175 with weak-prior mode enabled.

#SBATCH --job-name=DISC_LLM_RUBRIC_LLMRubric_225_25_9_175
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/AnnotationArena/imputer/ranking

export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -euo pipefail

SCRIPT_START=$SECONDS

SPLIT="LLMRubric_225_25_9_175"
DATA_DIR="DATA/LLM_RUBRIC/${SPLIT}"
DATA_BUNDLE="${DATA_DIR}/data_bundle.json"
OUTPUT_DIR="RESULTS/STAN/LLM_RUBRIC/DISCRETE"
RUN_NAME="${SPLIT}_DISCRETE_MISP_FLAT"

CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=500
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

write_temp_discrete_config() {
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
    "M": 6,
    "S": 6,
    "sigma_measurement": 0.5,
    "kappa": 2.0,
    "temperature": 1.0,
    "stan_type": "discrete",
}
with open(dst, "w") as f:
    json.dump({"datagen": merged}, f, indent=2)
PY
}

TMP_CONFIG=$(mktemp /tmp/llmrubric_discrete_config.XXXXXX.json)
trap 'rm -f "$TMP_CONFIG"' EXIT
write_temp_discrete_config "${DATA_DIR}/configs.json" "$TMP_CONFIG"

echo ""
echo "============================================================"
echo " LLM RUBRIC | Discrete Misspecified (flat-prior mode) | ${SPLIT}"
echo "  RUN_NAME      : ${RUN_NAME}"
echo "  M             : 6"
echo "  S             : 6"
echo "  CHAINS        : ${CHAINS}"
echo "  ITER_WARMUP   : ${ITER_WARMUP}"
echo "  ITER_SAMPLING : ${ITER_SAMPLING}"
echo "============================================================"

echo "[1/2] Running MCMC (discrete, weak-prior mode)..."
python STAN/stan_code/scripts/run_inference.py \
    --data-bundle                "${DATA_BUNDLE}" \
    --configs                    "${TMP_CONFIG}" \
    --output-dir                 "${OUTPUT_DIR}" \
    --run-name                   "${RUN_NAME}" \
    --stan-type                  "discrete" \
    --stan-file                  "STAN/stan_models/discrete_type_domain_model.stan" \
    --chains                     "${CHAINS}" \
    --iter-warmup                "${ITER_WARMUP}" \
    --iter-sampling              "${ITER_SAMPLING}" \
    --adapt-delta                "${ADAPT_DELTA}" \
    --max-treedepth              "${MAX_TREEDEPTH}" \
    --seed                       "${SEED}" \
    --transductive-use-test-observed \
    --stan-arg                   use_flat_priors=1 \
    --stan-arg                   N_pairwise_rankings=0 \
    --stan-arg                   N_missing_pairwise_rankings=0 \
    --overwrite-existing-data

echo "[2/2] Evaluating predictions..."
python STAN/stan_code/scripts/evaluate_predictions.py \
    --data-bundle        "${DATA_BUNDLE}" \
    --mcmc-dir           "${OUTPUT_DIR}/${RUN_NAME}" \
    --output-dir         "${OUTPUT_DIR}" \
    --run-name           "${RUN_NAME}_eval" \
    --csv-pattern        "discrete_type_domain_model-*.csv" \
    --overwrite-existing-data \
    --verbose

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " Done in $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " Results: ${OUTPUT_DIR}/${RUN_NAME}_eval"
echo "============================================================"
