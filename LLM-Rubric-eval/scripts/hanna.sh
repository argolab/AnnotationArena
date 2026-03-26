#!/bin/bash

#SBATCH -A jeisner1_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --job-name=hanna_tuning
#SBATCH --output=hanna%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Grid search script for hyperparameter tuning
source /home/hshi33/data_jeisner1/hshi33/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd /home/hshi33/data_jeisner1/hshi33/AnnotationArena/LLM-Rubric/scripts

# ============================================================================
# CONFIGURATION - Modify these paths
# ============================================================================
TRAIN_JSON="/home/hshi33/data_jeisner1/hshi33/AnnotationArena/src/input_hanna/data/initial_train.json"
TEST_JSON="/home/hshi33/data_jeisner1/hshi33/AnnotationArena/src/input_hanna/data/test.json"
BASE_OUTPUT_DIR="/home/hshi33/data_jeisner1/hshi33/AnnotationArena/LLM-Rubric/scripts/models_hanna"
JUDGE_MAP_DIR="/home/hshi33/data_jeisner1/hshi33/AnnotationArena/LLM-Rubric/experiments_hanna"

# Define hyperparameter grid
LAYER1_SIZES=(100 150)
LAYER2_SIZES=(100 150)
BATCH_SIZES=(32)
LEARNING_RATES=(0.01 0.005 0.001 0.0001)
PRETRAIN_EPOCHS=(500 1000 5000 8000)

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Create base directories
mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$JUDGE_MAP_DIR"

# Log file
LOG_FILE="$BASE_OUTPUT_DIR/grid_search.log"
RESULTS_FILE="$BASE_OUTPUT_DIR/grid_search_results.csv"

# Initialize results file with header
echo "experiment_id,layer1_size,layer2_size,batch_size,learning_rate,pretrain_epochs,train_time,train_status,overall_rmse,overall_pearson,overall_spearman,overall_kendall,overall_loss,eval_status" > "$RESULTS_FILE"

echo "Starting grid search with evaluation at $(date)" | tee -a "$LOG_FILE"
echo "Total combinations: $((${#LAYER1_SIZES[@]} * ${#LAYER2_SIZES[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#PRETRAIN_EPOCHS[@]}))" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Counter for experiments
EXPERIMENT_NUM=0
TOTAL_EXPERIMENTS=$((${#LAYER1_SIZES[@]} * ${#LAYER2_SIZES[@]} * ${#BATCH_SIZES[@]} * ${#LEARNING_RATES[@]} * ${#PRETRAIN_EPOCHS[@]}))

for layer1 in "${LAYER1_SIZES[@]}"; do
    for layer2 in "${LAYER2_SIZES[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for pretrain_ep in "${PRETRAIN_EPOCHS[@]}"; do
                    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
                    
                    # Create experiment ID
                    EXP_ID="exp_$(printf "%04d" $EXPERIMENT_NUM)_l1-${layer1}_l2-${layer2}_bs-${batch_size}_lr-${lr}_pt-${pretrain_ep}"
                    
                    # Create experiment directory
                    EXP_DIR="$BASE_OUTPUT_DIR/$EXP_ID"
                    mkdir -p "$EXP_DIR"
                    
                    # Log experiment start
                    echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] Starting: $EXP_ID" | tee -a "$LOG_FILE"
                    echo "  Parameters: layer1=$layer1, layer2=$layer2, batch_size=$batch_size, lr=$lr, pretrain=$pretrain_ep" | tee -a "$LOG_FILE"
                    
                    # Record start time
                    TRAIN_START_TIME=$(date +%s)
                    
                    # Run training with separate judge map for this experiment
                    JUDGE_MAP_PATH="$JUDGE_MAP_DIR/${EXP_ID}_judge_map.json"
                    
                    echo "  Running training..." | tee -a "$LOG_FILE"
                    python train_hanna.py \
                        --train-json "$TRAIN_JSON" \
                        --model-output-dir "$EXP_DIR" \
                        --judge-map "$JUDGE_MAP_PATH" \
                        --layer1-size "$layer1" \
                        --layer2-size "$layer2" \
                        --batch-size "$batch_size" \
                        --learning-rate "$lr" \
                        --pretraining-epochs "$pretrain_ep" \
                        --finetuning-epochs "$pretrain_ep" \
                        > "$EXP_DIR/train.log" 2>&1
                    
                    # Calculate training time
                    TRAIN_END_TIME=$(date +%s)
                    TRAIN_TIME=$((TRAIN_END_TIME - TRAIN_START_TIME))
                    
                    echo "  ✓ Training completed in ${TRAIN_TIME}s" | tee -a "$LOG_FILE"
                    
                    # Run evaluation
                    echo "  Running evaluation..." | tee -a "$LOG_FILE"
                    EVAL_OUTPUT_FILE="$EXP_DIR/eval_predictions.json"
                    
                    python hanna_predict.py \
                        --test-json "$TEST_JSON" \
                        --model-dir "$EXP_DIR" \
                        --judge-map "$JUDGE_MAP_PATH" \
                        --output-predictions "$EVAL_OUTPUT_FILE" \
                        --layer1-size "$layer1" \
                        --layer2-size "$layer2" \
                        --batch-size "$batch_size" \
                        --learning-rate "$lr" \
                        --pretraining-epochs "$pretrain_ep" \
                        --finetuning-epochs "$pretrain_ep" \
                        > "$EXP_DIR/eval.log" 2>&1
                    
                    # Extract metrics from evaluation output
                    EVAL_LOG="$EXP_DIR/eval.log"
                    OVERALL_RMSE=$(grep "Overall RMSE:" "$EVAL_LOG" | grep -oP '[0-9.]+' | head -1)
                    OVERALL_PEARSON=$(grep "Overall Pearson:" "$EVAL_LOG" | grep -oP '[0-9.]+' | head -1)
                    OVERALL_SPEARMAN=$(grep "Overall Spearman:" "$EVAL_LOG" | grep -oP '[0-9.]+' | head -1)
                    OVERALL_KENDALL=$(grep "Overall Kendall:" "$EVAL_LOG" | grep -oP '[0-9.]+' | head -1)
                    OVERALL_LOSS=$(grep "Overall Loss" "$EVAL_LOG" | grep -oP '[0-9.]+' | head -1)
                    
                    echo "  ✓ Evaluation completed" | tee -a "$LOG_FILE"
                    echo "    Pearson: $OVERALL_PEARSON, RMSE: $OVERALL_RMSE" | tee -a "$LOG_FILE"
                    
                    # Record results
                    echo "$EXP_ID,$layer1,$layer2,$batch_size,$lr,$pretrain_ep,$TRAIN_TIME,$OVERALL_RMSE,$OVERALL_PEARSON,$OVERALL_SPEARMAN,$OVERALL_KENDALL,$OVERALL_LOSS" >> "$RESULTS_FILE"
                    echo "" | tee -a "$LOG_FILE"
                    
                done
            done
        done
    done
done

echo "Grid search completed at $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$LOG_FILE"
echo "Total experiments: $TOTAL_EXPERIMENTS" | tee -a "$LOG_FILE"
echo "Training successful: $(grep -c ",success," "$RESULTS_FILE" || echo 0)" | tee -a "$LOG_FILE"
echo "Evaluation successful: $(grep -c ",success$" "$RESULTS_FILE" || echo 0)" | tee -a "$LOG_FILE"

# Show top 5 results by Pearson correlation
echo "" | tee -a "$LOG_FILE"
echo "Top 5 experiments by Pearson correlation:" | tee -a "$LOG_FILE"
grep ",success$" "$RESULTS_FILE" | sort -t',' -k11 -rn | head -5 | column -t -s',' | tee -a "$LOG_FILE"