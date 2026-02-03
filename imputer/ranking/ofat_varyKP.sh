#!/bin/bash
# OFAT: Vary κ (Kappa) Experiments Only

# set -e  # Exit on error

# Fixed base parameters for all experiments
BASE_I=10
BASE_J=12
BASE_C=5
BASE_K_TRAIN=10
BASE_K_TEST=10

# Baseline hyperparameter values (center of OFAT space)
BASE_D=16
BASE_SIGMA_ANNOTATOR=0.5
BASE_SIGMA_MEASUREMENT=0.1
BASE_KAPPA=10
BASE_PROTOCOL="tie_breaking"  # SMAR
BASE_PROTOCOL_CODE="smar"
BASE_USE_CONCAT=0  # 0 = disabled, 1 = enabled for AtomCompositional embeddings

# Marformer hyperparameters (fixed across all experiments)
MARFORMER_EPOCHS=70
MARFORMER_LR=1e-4
MARFORMER_MASKING_RATE=0.15
MARFORMER_MASKED_LOSS_WEIGHT=15
MARFORMER_OBSERVED_LOSS_WEIGHT=1
MARFORMER_MASK_AUGMENTATIONS=5
MARFORMER_EARLY_STOPPING_PATIENCE=10
MARFORMER_DEVICE="cuda"

# Stan hyperparameters - TWO configurations per experiment
STAN_4C_CHAINS=4
STAN_1C_CHAINS=1
STAN_1C_ITER_SAMPLING=1500
STAN_1C_WARMUP=500
STAN_4C_ITER_SAMPLING=1000
STAN_4C_WARMUP=1000

# Function to run complete experiment pipeline
run_experiment() {
    local varied_param=$1
    local D=$2
    local sigma_annotator=$3
    local sigma_measurement=$4
    local kappa=$5
    local protocol=$6
    local protocol_code=$7
    local use_concat_embedding=$8

    # Format values for folder naming (remove decimals for cleaner names)
    local d_str=$D
    local sa_str=$(echo $sigma_annotator | sed 's/\.//g')
    local sm_str=$(echo $sigma_measurement | sed 's/\.//g')
    local kp_str=$(echo $kappa | sed 's/\.//g')
    local uc_str=$use_concat_embedding

    # Construct run name
    local run_name="ofat_${varied_param}_D${d_str}_sa${sa_str}_sm${sm_str}_kp${kp_str}_uc${uc_str}_${protocol_code}"

    echo ""
    echo "=========================================="
    echo "EXPERIMENT: $run_name"
    echo "  D=$D, σ_ann=$sigma_annotator, σ_meas=$sigma_measurement, κ=$kappa, protocol=$protocol, use_concat_embedding=$use_concat_embedding"
    echo "=========================================="
    echo ""

    # Determine protocol-specific arguments
    local protocol_args=""
    if [ "$protocol" == "extended_rankings" ]; then
        protocol_args="--extended-pairwise-rate 0.2"
    elif [ "$protocol" == "mcar" ]; then
        protocol_args="--mcar-missing-rate 0.5"
    fi

    # Toggle for concat-based AtomCompositional embeddings
    local concat_flag=""
    if [ "$use_concat_embedding" == "1" ]; then
        concat_flag="--use-concat-embedding"
    fi

    # Step 1: Generate data
    echo "[Step 1/6] Generating data..."
    python stan/scripts/generate_data.py \
        --K-train $BASE_K_TRAIN \
        --K-test $BASE_K_TEST \
        --I $BASE_I \
        --J $BASE_J \
        --D $D \
        --C $BASE_C \
        --observation-protocol $protocol \
        --sigma-annotator $sigma_annotator \
        --sigma-measurement $sigma_measurement \
        --kappa $kappa \
        --run-name $run_name \
        --overwrite-existing-data \
        $protocol_args </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Data generation failed for $run_name"
        return 1
    fi

    # Step 2: Run Marformer
    echo "[Step 2/6] Running Marformer..."
    python imputer/run_imputer.py \
        --data-dir OUTPUT/generated_data/${run_name} \
        --run-name ${run_name}_marformer \
        --overwrite-existing-data \
        --embedding-dim $((D * 3)) \
        --epochs $MARFORMER_EPOCHS \
        --lr $MARFORMER_LR \
        --masking-rate $MARFORMER_MASKING_RATE \
        --masked-loss-weight $MARFORMER_MASKED_LOSS_WEIGHT \
        --observed-loss-weight $MARFORMER_OBSERVED_LOSS_WEIGHT \
        --mask-augmentations $MARFORMER_MASK_AUGMENTATIONS \
        --early-stopping \
        --early-stopping-metric loss \
        --early-stopping-patience $MARFORMER_EARLY_STOPPING_PATIENCE \
        --transductive_learning \
        --no-final-norm \
        --normalize-parameter \
        --full_random \
        --device $MARFORMER_DEVICE \
        $concat_flag </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Marformer training failed for $run_name"
        return 1
    fi

    # Step 3: Run Stan inference (4 chains)
    echo "[Step 3/6] Running Stan inference (4 chains)..."
    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --chains $STAN_4C_CHAINS \
        --iter-sampling $STAN_4C_ITER_SAMPLING \
        --iter-warmup $STAN_4C_WARMUP \
        --run-name ${run_name}_stan4c \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan inference (4 chains) failed for $run_name"
        return 1
    fi

    # Step 4: Run Stan inference (1 chain, long)
    echo "[Step 4/6] Running Stan inference (1 chain, long)..."
    python stan/scripts/run_inference.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --chains $STAN_1C_CHAINS \
        --iter-sampling $STAN_1C_ITER_SAMPLING \
        --iter-warmup $STAN_1C_WARMUP \
        --run-name ${run_name}_stan1c \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan inference (1 chain) failed for $run_name"
        return 1
    fi

    # Step 5: Evaluate Stan predictions (4-chain version)
    echo "[Step 5/6] Evaluating Stan predictions (4 chains)..."
    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${run_name}_stan4c \
        --run-name ${run_name}_stan4c_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan evaluation (4 chains) failed for $run_name"
        return 1
    fi

    # Step 5b: Evaluate Stan predictions (1-chain version)
    echo "[Step 5b/6] Evaluating Stan predictions (1 chain)..."
    python stan/scripts/evaluate_predictions.py \
        --data-bundle OUTPUT/generated_data/${run_name}/data_bundle.json \
        --mcmc-dir OUTPUT/domain_model/runs/${run_name}_stan1c \
        --run-name ${run_name}_stan1c_eval \
        --overwrite-existing-data </dev/null

    if [ $? -ne 0 ]; then
        echo "ERROR: Stan evaluation (1 chain) failed for $run_name"
        return 1
    fi

    # Step 6: Generate visualization plots
    echo "[Step 6/6] Generating visualization plots..."
    python utils/visualize.py \
        --run-dir OUTPUT/IMPUTER/${run_name}_marformer \
        --stan-metrics OUTPUT/domain_model/eval/${run_name}_stan4c_eval/predictive_metrics.json </dev/null

    if [ $? -ne 0 ]; then
        echo "WARNING: Visualization failed for $run_name (continuing anyway)"
    else
        echo "  - Plots saved to OUTPUT/IMPUTER/${run_name}_marformer/plots/"
    fi

    echo ""
    echo "✓ COMPLETED: $run_name"
    echo ""
}

# Main execution
echo "=============================================="
echo "OFAT: VARY κ (KAPPA) EXPERIMENTS"
echo "=============================================="
echo ""
echo "Base parameters:"
echo "  I=$BASE_I, J=$BASE_J, C=$BASE_C"
echo "  K_train=$BASE_K_TRAIN, K_test=$BASE_K_TEST"
echo ""
echo "Center (baseline) hyperparameters:"
echo "  D=$BASE_D"
echo "  σ_annotator=$BASE_SIGMA_ANNOTATOR"
echo "  σ_measurement=$BASE_SIGMA_MEASUREMENT"
echo "  κ=$BASE_KAPPA (baseline, not run in this script)"
echo "  Protocol=$BASE_PROTOCOL (SMAR)"
echo "  use_concat_embedding=$BASE_USE_CONCAT"
echo ""
echo "Total experiments: 2"
echo "  Vary κ: 2 (0.5, 2.0)"
echo ""
echo "Each experiment runs:"
echo "  - 1 Marformer training"
echo "  - 1 Stan (4 chains)"
echo "  - 1 Stan (1 chain, 1500 samples)"
echo "  - 1 Visualization (learning curves + calibration)"
echo "=============================================="
echo ""

# Track experiment count
TOTAL_EXPERIMENTS=0
FAILED_EXPERIMENTS=0

# ===== OFAT: Vary kappa =====
echo "======================================"
echo "OFAT SET: Varying κ"
echo "======================================"
for KP in 2.0 5.0 20.0 50.0; do
    run_experiment "varyKP" $BASE_D $BASE_SIGMA_ANNOTATOR $BASE_SIGMA_MEASUREMENT $KP $BASE_PROTOCOL $BASE_PROTOCOL_CODE $BASE_USE_CONCAT
    if [ $? -ne 0 ]; then
        ((FAILED_EXPERIMENTS++))
    fi
    ((TOTAL_EXPERIMENTS++))
done

# ===== Summary =====
echo ""
echo "=============================================="
echo "OFAT VARY κ EXPERIMENTS COMPLETE"
echo "=============================================="
echo "Total experiments run: $TOTAL_EXPERIMENTS"
echo "Failed experiments: $FAILED_EXPERIMENTS"
echo "Successful experiments: $((TOTAL_EXPERIMENTS - FAILED_EXPERIMENTS))"
echo ""
echo "Results saved in:"
echo "  - Data: OUTPUT/generated_data/ofat_varyKP_*"
echo "  - Marformer: OUTPUT/IMPUTER/ofat_varyKP_*_marformer"
echo "  - Stan (4c): OUTPUT/domain_model/runs/ofat_varyKP_*_stan4c"
echo "  - Stan (4c) Eval: OUTPUT/domain_model/eval/ofat_varyKP_*_stan4c_eval"
echo "  - Stan (1c): OUTPUT/domain_model/runs/ofat_varyKP_*_stan1c"
echo "  - Stan (1c) Eval: OUTPUT/domain_model/eval/ofat_varyKP_*_stan1c_eval"
echo "=============================================="



