#!/bin/bash
# Imputer ablation sweep on LLMRubric real data.
# Copies run_real_data.sh and adds leave-one-out ablation sweep + summarization/plotting.
#
# Two variants (controlled by BUNDLE):
#   hard  → OUTPUT/generated_data/llm_rubric       (hard labels, one-hot CE)
#   dist  → OUTPUT/generated_data/llm_rubric_dist  (soft labels, LLM distributions)
#
# Usage (from repo root):
#   bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh
#   BUNDLE=dist bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh
#   DRY_RUN=1 bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh  # echo commands only
#   SMOKE_RUN=1 bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh  # 5 epochs only, quick validation

set -e

# ── Bundle variant ────────────────────────────────────────────────────────────
BUNDLE="${BUNDLE:-hard}"

if [ "$BUNDLE" == "dist" ]; then
    DATA_DIR="OUTPUT/generated_data/llm_rubric_dist"
    ABLATION_PREFIX="llm_rubric_marformer_dist_ablation"
else
    DATA_DIR="OUTPUT/generated_data/llm_rubric"
    ABLATION_PREFIX="llm_rubric_marformer_ablation"
fi

# ── Smoke run: 5 epochs only, separate output prefix ─────────────────────────────
if [ "${SMOKE_RUN:-0}" = "1" ]; then
    EPOCHS=5
    ABLATION_PREFIX="${ABLATION_PREFIX}_smoke"
    echo "[SMOKE_RUN] Training 5 epochs per ablation, prefix=${ABLATION_PREFIX}"
else
    EPOCHS=100
fi

# ── Marformer hyperparameters (identical to run_real_data.sh) ────────────────────
EMBEDDING_DIM=72
W_INIT="identity"
ENCODER_LAYERS=4
ATTENTION_HEADS=4
NUM_FFN_LAYERS=1
D_FF=128
DROPOUT=0.1
EMBEDDING_DROPOUT=0.5
WEIGHT_DECAY=0.01
LR=2e-4
MASKING_RATE=0.15
MASKED_LOSS_WEIGHT=15
OBSERVED_LOSS_WEIGHT=1
MASK_AUGMENTATIONS=5
MAX_ITEM=10
BATCH_SIZE=1
GRADIENT_CLIP_VAL=0.0
DEVICE="cpu"
DEVICES=1

USE_COSINE_SCHEDULE=true
WARMUP_STEPS=5

# ── Derived flags ─────────────────────────────────────────────────────────────
cosine_flags=""
if [ "$USE_COSINE_SCHEDULE" == "true" ]; then
    cosine_flags="--use-cosine-schedule --warmup-steps $WARMUP_STEPS"
fi

# ── Ablation configs: id -> extra CLI flags ─────────────────────────────────────
# BASE = no ablations (original Imputer)
ablation_ids=(BASE no_dropout no_mask_aug no_rand_mask no_cosine no_transductive no_pointer no_param_norm no_human_sup no_wd)

get_ablation_flags() {
    local id=$1
    case "$id" in
        BASE) echo "" ;;
        no_dropout) echo "--ablation-no-dropout" ;;
        no_mask_aug) echo "--ablation-no-mask-augmentation" ;;
        no_rand_mask) echo "--ablation-no-random-masking" ;;
        no_cosine) echo "--ablation-no-cosine-schedule" ;;
        no_transductive) echo "--ablation-no-transductive" ;;
        no_pointer) echo "--ablation-no-pointer-mechanism" ;;
        no_param_norm) echo "--ablation-no-parameter-normalization" ;;
        no_human_sup) echo "--ablation-no-human-supervision" ;;
        no_wd) echo "--ablation-no-weight-decay" ;;
        *) echo "Unknown ablation: $id" >&2; exit 1 ;;
    esac
}

# ── Run sweep ──────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "LLMRubric Real Data — Imputer Ablation Sweep"
echo "  bundle:   $BUNDLE"
echo "  data_dir: $DATA_DIR"
echo "  prefix:   $ABLATION_PREFIX"
echo "  ablations: ${ablation_ids[*]}"
echo "=========================================="
echo ""

for ablation_id in "${ablation_ids[@]}"; do
    EXTRA_FLAGS=$(get_ablation_flags "$ablation_id")
    RUN_NAME="${ABLATION_PREFIX}_${ablation_id}"

    echo ""
    echo "----------------------------------------"
    echo "[Ablation] $ablation_id → $RUN_NAME"
    echo "  extra: $EXTRA_FLAGS"
    echo "----------------------------------------"

    CMD="python imputer/run_imputer.py \
        --data-dir $DATA_DIR \
        --run-name $RUN_NAME \
        --overwrite-existing-data \
        --embedding-dim $EMBEDDING_DIM \
        --encoder-layers $ENCODER_LAYERS \
        --attention-heads $ATTENTION_HEADS \
        --num_ffn_layers $NUM_FFN_LAYERS \
        --d-ff $D_FF \
        --dropout $DROPOUT \
        --weight-decay $WEIGHT_DECAY \
        --epochs $EPOCHS \
        --lr $LR \
        --masking-rate $MASKING_RATE \
        --masked-loss-weight $MASKED_LOSS_WEIGHT \
        --observed-loss-weight $OBSERVED_LOSS_WEIGHT \
        --mask-augmentations $MASK_AUGMENTATIONS \
        --max-item $MAX_ITEM \
        --batch-size $BATCH_SIZE \
        --gradient-clip-val $GRADIENT_CLIP_VAL \
        --no-final-norm \
        --normalize-parameter \
        --device $DEVICE \
        --devices $DEVICES \
        --save-model-every 5 \
        --llm-annotator-id 24 \
        --human-observed-rate 0.2 \
        --item-embedding-dropout $EMBEDDING_DROPOUT \
        --w-init $W_INIT \
        --loss-fn kl \
        --transductive_learning \
        $cosine_flags \
        $EXTRA_FLAGS"

    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[DRY_RUN] $CMD"
    else
        eval $CMD
    fi
done

# ── Post-sweep: summarization and plotting ────────────────────────────────────
echo ""
echo "=========================================="
echo "Post-sweep: summarization and plotting"
echo "=========================================="

OUTPUT_ROOT="OUTPUT/IMPUTER"
PLOT_DIR="${OUTPUT_ROOT}/plots/IMPUTER_ABLATION"
mkdir -p "$PLOT_DIR"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY_RUN] python scripts/real_data/summarize_imputer_ablation.py --output-root $OUTPUT_ROOT --run-prefix $ABLATION_PREFIX"
    echo "[DRY_RUN] python scripts/real_data/plot_imputer_ablation_curves.py --output-root $OUTPUT_ROOT --run-prefix $ABLATION_PREFIX --plot-dir $PLOT_DIR"
else
    python scripts/real_data/summarize_imputer_ablation.py \
        --output-root "$OUTPUT_ROOT" \
        --run-prefix "$ABLATION_PREFIX"

    python scripts/real_data/plot_imputer_ablation_curves.py \
        --output-root "$OUTPUT_ROOT" \
        --run-prefix "$ABLATION_PREFIX" \
        --plot-dir "$PLOT_DIR"
fi

echo ""
echo "✓ Ablation sweep complete"
echo "  Summary table: $OUTPUT_ROOT/llm_rubric_imputer_ablation_summary.csv"
echo "  Plots: $PLOT_DIR/"
echo ""
