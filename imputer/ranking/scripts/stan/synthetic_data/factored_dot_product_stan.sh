#!/bin/bash
# Easy-data axis invariance: IJK mode (hold_I=0, hold_J=0, hold_K=0)
# Full dependence on all three axes (baseline complex case).

set -e

# Fixed base parameters (mirrors OFAT center setup where possible)
BASE_I=9
BASE_J=25
BASE_C=4
BASE_K_TRAIN=50
BASE_K_TEST=25

# Baseline hyperparameter values
BASE_D=8
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10
BASE_PROTOCOL="mcar"  # SMAR
BASE_PROTOCOL_CODE="mcar"


DISABLE_RANKING=1

ranking_args=""

if [ $DISABLE_RANKING == 0 ]; then
    ranking_args="--enable-pairwise-rankings"
fi


# Format values for folder naming
d_str=$BASE_D
sa_str=$(echo $BASE_SIGMA_ANNOTATOR | sed 's/\.//g')
sm_str=$(echo $BASE_SIGMA_MEASUREMENT | sed 's/\.//g')
kp_str=$(echo $BASE_KAPPA | sed 's/\.//g')

# Construct run name
run_name="K_train_${BASE_K_TRAIN}_K_test_${BASE_K_TEST}_I_${BASE_I}_J_${BASE_J}_factored_dot_product"

#run_name="normal_noise_dot_product_llm_rubric"

# Determine protocol-specific arguments
protocol_args=""
if [ "$BASE_PROTOCOL" == "extended_rankings" ]; then
    protocol_args="--extended-pairwise-rate 0.2"
elif [ "$BASE_PROTOCOL" == "mcar" ]; then
    protocol_args="--mcar-missing-rate 0.5"
fi

# Step 1: Generate data (Stan)
# echo "[Step 1/6] Generating data..."

# python stan/scripts/generate_data.py \
#     --K-train $BASE_K_TRAIN \
#     --K-test $BASE_K_TEST \
#     --I $BASE_I \
#     --J $BASE_J \
#     --D $BASE_D \
#     --C $BASE_C \
#     --observation-protocol $BASE_PROTOCOL \
#     --sigma-annotator $BASE_SIGMA_ANNOTATOR \
#     --sigma-measurement $BASE_SIGMA_MEASUREMENT \
#     --kappa $BASE_KAPPA \
#     --run-name $run_name \
#     --overwrite-existing-data \
#     --stan-type factored-dot-product \
#     $ranking_args \
#     $protocol_args </dev/null

# if [ $? -ne 0 ]; then
#     echo "ERROR: Data generation failed for $run_name"
#     exit 1
# fi

# ── Dataset (always llmrubric_dist) ───────────────────────────────────────────
DATA_DIR="OUTPUT/generated_data/${run_name}"

# ── Dirichlet hyperparameter ──────────────────────────────────────────────────
# Larger  → LLM observations are treated as more reliable (tighter Dirichlet)
# Smaller → LLM observations are treated as more noisy
ALPHA_LLM="${ALPHA_LLM:-5.0}"

# Include alpha in folder name so sweeps don't collide
RUN_BASE="${run_name}"

# ── MCMC hyperparameters ──────────────────────────────────────────────────────
CHAINS="${CHAINS:-1}"
ITER_WARMUP="${ITER_WARMUP:-100}"
ITER_SAMPLING="${ITER_SAMPLING:-300}"
ADAPT_DELTA="${ADAPT_DELTA:-0.85}"
MAX_TREEDEPTH="${MAX_TREEDEPTH:-12}"
SEED="${SEED:-42}"

# ── Stan model hyperparameters ────────────────────────────────────────────────
EMBEDDING_DIM="${EMBEDDING_DIM:-8}"

echo ""
echo "=================================================="
echo "Stan Dirichlet Real-Data Inference (LLMRubric)"
echo "  data_dir:       $DATA_DIR"
echo "  run_base:       $RUN_BASE"
echo "  alpha_llm:      $ALPHA_LLM"
echo "  chains:         $CHAINS  warmup=$ITER_WARMUP  sampling=$ITER_SAMPLING"
echo "  adapt_delta:    $ADAPT_DELTA  max_treedepth=$MAX_TREEDEPTH"
echo "  D=$EMBEDDING_DIM"
echo "=================================================="
echo ""

# ── Step 1: MCMC inference with Dirichlet model ───────────────────────────────
echo "[Step 1/2] Running Stan MCMC (Dirichlet model)..."
python stan/scripts/run_inference.py \
    --data-bundle    "$DATA_DIR/data_bundle.json" \
    --run-name       "$RUN_BASE" \
    --chains         "$CHAINS" \
    --iter-warmup    "$ITER_WARMUP" \
    --iter-sampling  "$ITER_SAMPLING" \
    --adapt-delta    "$ADAPT_DELTA" \
    --max-treedepth  "$MAX_TREEDEPTH" \
    --seed           "$SEED" \
    --override-D     "$EMBEDDING_DIM" \
    --alpha-llm      "$ALPHA_LLM" \
    --stan-type      "factored-dot-product" \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference failed"
    exit 1
fi

# ── Step 2: Evaluate predictions ──────────────────────────────────────────────
echo ""
echo "[Step 2/2] Evaluating Stan predictions..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle  "$DATA_DIR/data_bundle.json" \
    --mcmc-dir     "OUTPUT/domain_model/runs/$RUN_BASE" \
    --run-name     "${RUN_BASE}_eval" \
    --csv-pattern  "normal_noise_dot_product_model-*.csv" \
    --overwrite-existing-data \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation failed"
    exit 1
fi

RUN_BASE="${run_name}_normal"

# ── Step 1: MCMC inference with Dirichlet model ───────────────────────────────
echo "[Step 1/2] Running Stan MCMC (Dirichlet model)..."
python stan/scripts/run_inference.py \
    --data-bundle    "$DATA_DIR/data_bundle.json" \
    --run-name       "$RUN_BASE" \
    --chains         "$CHAINS" \
    --iter-warmup    "$ITER_WARMUP" \
    --iter-sampling  "$ITER_SAMPLING" \
    --adapt-delta    "$ADAPT_DELTA" \
    --max-treedepth  "$MAX_TREEDEPTH" \
    --seed           "$SEED" \
    --override-D     "$EMBEDDING_DIM" \
    --alpha-llm      "$ALPHA_LLM" \
    --stan-type      "normal-noise-dot-product" \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference failed"
    exit 1
fi

# ── Step 2: Evaluate predictions ──────────────────────────────────────────────
echo ""
echo "[Step 2/2] Evaluating Stan predictions..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle  "$DATA_DIR/data_bundle.json" \
    --mcmc-dir     "OUTPUT/domain_model/runs/$RUN_BASE" \
    --run-name     "${RUN_BASE}_eval" \
    --csv-pattern  "normal_noise_dot_product_model-*.csv" \
    --overwrite-existing-data \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation failed"
    exit 1
fi

RUN_BASE="${run_name}_tensor"

# ── Step 1: MCMC inference with Dirichlet model ───────────────────────────────
echo "[Step 1/2] Running Stan MCMC (Dirichlet model)..."
python stan/scripts/run_inference.py \
    --data-bundle    "$DATA_DIR/data_bundle.json" \
    --run-name       "$RUN_BASE" \
    --chains         "$CHAINS" \
    --iter-warmup    "$ITER_WARMUP" \
    --iter-sampling  "$ITER_SAMPLING" \
    --adapt-delta    "$ADAPT_DELTA" \
    --max-treedepth  "$MAX_TREEDEPTH" \
    --seed           "$SEED" \
    --override-D     "$EMBEDDING_DIM" \
    --alpha-llm      "$ALPHA_LLM" \
    --stan-type      "tensor" \
    --overwrite-existing-data

if [ $? -ne 0 ]; then
    echo "ERROR: Stan inference failed"
    exit 1
fi

# ── Step 2: Evaluate predictions ──────────────────────────────────────────────
echo ""
echo "[Step 2/2] Evaluating Stan predictions..."
python stan/scripts/evaluate_predictions.py \
    --data-bundle  "$DATA_DIR/data_bundle.json" \
    --mcmc-dir     "OUTPUT/domain_model/runs/$RUN_BASE" \
    --run-name     "${RUN_BASE}_eval" \
    --csv-pattern  "tensor_model-*.csv" \
    --overwrite-existing-data \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Stan evaluation failed"
    exit 1
fi