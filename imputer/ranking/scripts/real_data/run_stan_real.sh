#!/bin/bash
# Run Stan on a real annotation dataset (HANNA or LLMRubric).
#
# Controls (set as env vars or edit below):
#   DATASET  — "hanna" | "llmrubric"   (default: hanna)
#   BUNDLE   — "hard"  | "dist"        (default: hard)
#
# Usage (from repo root):
#   bash scripts/real_data/run_stan_real.sh
#   DATASET=hanna      BUNDLE=dist  bash scripts/real_data/run_stan_real.sh
#   DATASET=llmrubric  BUNDLE=hard  bash scripts/real_data/run_stan_real.sh
#   DATASET=llmrubric  BUNDLE=dist  bash scripts/real_data/run_stan_real.sh

set -e

# ── Dataset & bundle ──────────────────────────────────────────────────────────
DATASET="${DATASET:-hanna}"
BUNDLE="${BUNDLE:-hard}"

if [ "$DATASET" == "hanna" ]; then
    if [ "$BUNDLE" == "dist" ]; then
        DATA_DIR="OUTPUT/generated_data/hanna_dist"
        RUN_BASE="hanna_stan_dist"
    else
        DATA_DIR="OUTPUT/generated_data/hanna"
        RUN_BASE="hanna_stan_hard"
    fi
elif [ "$DATASET" == "llmrubric" ]; then
    if [ "$BUNDLE" == "dist" ]; then
        DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
        RUN_BASE="llmrubric_stan_dist"
    else
        DATA_DIR="OUTPUT/generated_data/llm_rubric"
        RUN_BASE="llmrubric_stan_hard"
    fi
else
    echo "ERROR: DATASET must be 'hanna' or 'llmrubric', got: $DATASET"
    exit 1
fi

# ── MCMC hyperparameters ──────────────────────────────────────────────────────
CHAINS=1
ITER_WARMUP=300
ITER_SAMPLING=100
ADAPT_DELTA=0.85
MAX_TREEDEPTH=12
SEED=42

# ── Stan model hyperparameters (defaults written to configs.json) ─────────────
# Override any individual param with --override-* flags below if needed.
EMBEDDING_DIM=8

echo ""
echo "=================================================="
echo "Stan Real-Data Inference"
echo "  dataset:   $DATASET"
echo "  bundle:    $BUNDLE"
echo "  data_dir:  $DATA_DIR"
echo "  run_base:  $RUN_BASE"
echo "  chains:    $CHAINS  warmup=$ITER_WARMUP  sampling=$ITER_SAMPLING"
echo "  D=$EMBEDDING_DIM"
echo "=================================================="
echo ""

# ── Dist flag ─────────────────────────────────────────────────────────────────
dist_flag=""
if [ "$BUNDLE" == "dist" ]; then
    dist_flag="--use-dist"
fi

# ── Step 1: Run MCMC inference ────────────────────────────────────────────────
echo "[Step 1/2] Running Stan MCMC..."
python stan/scripts/run_inference.py \
    --data-bundle  "$DATA_DIR/data_bundle.json" \
    --run-name     "$RUN_BASE" \
    --chains       "$CHAINS" \
    --iter-warmup  "$ITER_WARMUP" \
    --iter-sampling "$ITER_SAMPLING" \
    --adapt-delta  "$ADAPT_DELTA" \
    --max-treedepth "$MAX_TREEDEPTH" \
    --seed         "$SEED" \
    --stan-arg     "D=$EMBEDDING_DIM" \
    --overwrite-existing-data \
    $dist_flag

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference failed"
    exit 1
fi

# ── Step 2: Evaluate predictions ──────────────────────────────────────────────
CSV_PATTERN_FLAG=""
if [ "$BUNDLE" == "dist" ]; then
    CSV_PATTERN_FLAG='--csv-pattern stan_dist_model-*.csv'
fi

echo ""
echo "[Step 2/2] Evaluating Stan predictions..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle  $DATA_DIR/data_bundle.json \
    --mcmc-dir     OUTPUT/domain_model/runs/$RUN_BASE \
    --run-name     ${RUN_BASE}_eval \
    $CSV_PATTERN_FLAG \
    --overwrite-existing-data \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation failed"
    exit 1
fi

echo ""
echo "Done: OUTPUT/domain_model/eval/${RUN_BASE}_eval"
echo ""
