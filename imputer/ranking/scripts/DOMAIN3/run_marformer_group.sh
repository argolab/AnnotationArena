#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

DOMAIN3_DEFAULT_CLUSTER_ROOT="/home/psingh54/scratchjeisner1/psingh54/Marformer/imputer/ranking"
DOMAIN3_DEFAULT_LOCAL_ROOT="/Users/prabhavsingh/Documents/JHU/JHUResearch/Marformer/imputer/ranking"

if [[ -n "${DOMAIN3_REPO_ROOT_OVERRIDE:-}" && -f "${DOMAIN3_REPO_ROOT_OVERRIDE}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_REPO_ROOT_OVERRIDE}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "${DOMAIN3_DEFAULT_CLUSTER_ROOT}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_CLUSTER_ROOT}"
elif [[ -f "${DOMAIN3_DEFAULT_LOCAL_ROOT}/scripts/DOMAIN3/common.sh" ]]; then
    DOMAIN3_REPO_ROOT="${DOMAIN3_DEFAULT_LOCAL_ROOT}"
else
    DOMAIN3_REPO_ROOT="$(pwd)"
fi

export DOMAIN3_REPO_ROOT_OVERRIDE="${DOMAIN3_REPO_ROOT}"
# shellcheck disable=SC1091
source "${DOMAIN3_REPO_ROOT}/scripts/DOMAIN3/common.sh"

FAMILY_LABEL=""
OUTPUT_ROOT=""
AXIS=""
PROTOCOL=""
SIZES_CSV=""
USE_EMBED_UNEMBED=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family-label)
            FAMILY_LABEL="$2"
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
        --protocol)
            PROTOCOL="$2"
            shift 2
            ;;
        --sizes)
            SIZES_CSV="$2"
            shift 2
            ;;
        --use-embed-unembed)
            USE_EMBED_UNEMBED=true
            shift 1
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${FAMILY_LABEL}" || -z "${OUTPUT_ROOT}" || -z "${AXIS}" || -z "${PROTOCOL}" || -z "${SIZES_CSV}" ]]; then
    echo "Missing required arguments to run_marformer_group.sh" >&2
    exit 1
fi

if [[ "${AXIS}" == "annotator" ]]; then
    AXIS="annot"
fi

if [[ "${AXIS}" != "item" && "${AXIS}" != "annot" ]]; then
    echo "Unsupported axis for run_marformer_group.sh: ${AXIS}" >&2
    exit 1
fi

domain3_setup_env
DATASET_LABEL="${DOMAIN3_DATASET_LABEL:-DOMAIN3-FINAL}"
DATA_ROOT="${DOMAIN3_DATA_ROOT:-DATA/STAN/DOMAIN3-FINAL}"
SPLIT_PREFIX="${DOMAIN3_SPLIT_PREFIX:-DOMAIN3-FINAL}"
EXP_PREFIX="${DOMAIN3_EXP_PREFIX:-domain3_final}"
SIZES_TAG="${SIZES_CSV//,/__}"
FAMILY_LABEL_LOWER="$(domain3_to_lower "${FAMILY_LABEL}")"
EXP_NAME="${EXP_PREFIX}_${FAMILY_LABEL_LOWER}_${AXIS}_${PROTOCOL}_${SIZES_TAG}"
EXP_NAME="${EXP_NAME// /_}"
domain3_enable_file_logging "${EXP_NAME}"
domain3_enter_repo

SCRIPT_START=$SECONDS

SEED=42
TYPE_EMBEDDING_INIT="kaiming"
EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
NUM_FFN_LAYERS=1
DROPOUT=0.1
EPOCHS=300
LR=2e-4
LR_SCHEDULE="none"
LR_MIN=1e-5
WEIGHT_DECAY=0.01
MASKING_RATE=0.15
MASK_AUGMENTATIONS=5
MASKED_LOSS_WEIGHT=15.0
OBSERVED_LOSS_WEIGHT=1.0
DEVICE="$(domain3_choose_torch_device)"
MAX_ITEM="${DOMAIN3_MAX_ITEM:-100}"
ITEM_DROPOUT_RATE=1.0
ANNOTATOR_DROPOUT_RATE=0.0
ITEM_REG_WEIGHT=0.0
ATTRIBUTE_REG_WEIGHT=0.0
ANNOTATOR_REG_WEIGHT=0.0
USE_PER_HEAD_REL=false
SCALE_SHARED_REL=true
USE_POINTER=true
USE_REL_VALUE=false
USE_ADDONE_ATTN=false
USE_DEVIATION_NORM=false
USE_GRAPH_MASK=false
LLM_INPUT_DIST=false
OVERWRITE_EXISTING=true

if [[ "${AXIS}" == "annot" ]]; then
    ANNOTATOR_REG_WEIGHT=1e-3
    ITEM_DROPOUT_RATE=0.0
    ANNOTATOR_DROPOUT_RATE=0.8
    ITEM_REG_WEIGHT=0.0
    ATTRIBUTE_REG_WEIGHT=0.0
fi

PER_HEAD_FLAG=""
[ "${USE_PER_HEAD_REL}" = "false" ] && PER_HEAD_FLAG="--no-per-head-rel"
SCALE_FLAG=""
[ "${SCALE_SHARED_REL}" = "true" ] && SCALE_FLAG="--scale-shared-rel"
POINTER_FLAG=""
[ "${USE_POINTER}" = "true" ] && POINTER_FLAG="--use-pointer"
REL_VALUE_FLAG=""
[ "${USE_REL_VALUE}" = "true" ] && REL_VALUE_FLAG="--use-rel-value"
ADDONE_FLAG=""
[ "${USE_ADDONE_ATTN}" = "true" ] && ADDONE_FLAG="--use-addone-attn"
DEVNORM_FLAG=""
[ "${USE_DEVIATION_NORM}" = "true" ] && DEVNORM_FLAG="--use-deviation-norm"
GRAPHMASK_FLAG=""
[ "${USE_GRAPH_MASK}" = "true" ] && GRAPHMASK_FLAG="--use-graph-mask"
LLM_DIST_FLAG=""
[ "${LLM_INPUT_DIST}" = "true" ] && LLM_DIST_FLAG="--llm-input-dist"
OVERWRITE_FLAG=""
[ "${OVERWRITE_EXISTING}" = "true" ] && OVERWRITE_FLAG="--overwrite-existing-data"
TRANSDUCTIVE_FLAG=""
[ "${PROTOCOL}" = "transductive" ] && TRANSDUCTIVE_FLAG="--transductive-learning"
EMBED_UNEMBED_FLAG=""
[ "${USE_EMBED_UNEMBED}" = "true" ] && EMBED_UNEMBED_FLAG="--use-embed-unembed"

IFS=',' read -r -a SIZE_LIST <<< "${SIZES_CSV}"

domain3_print_block "${DATASET_LABEL} | ${FAMILY_LABEL} | ${AXIS} | ${PROTOCOL}"
echo " OUTPUT_ROOT         : ${OUTPUT_ROOT}"
echo " REPO_ROOT           : ${DOMAIN3_REPO_ROOT}"
echo " LOG_FILE            : ${DOMAIN3_LOG_FILE}"
echo " DATA_ROOT           : ${DATA_ROOT}"
echo " SPLIT_PREFIX        : ${SPLIT_PREFIX}"
echo " SIZES               : ${SIZE_LIST[*]}"
echo " EMBEDDING_DIM       : ${EMBEDDING_DIM}"
echo " NUM_LAYERS          : ${NUM_LAYERS}"
echo " ATTENTION_HEADS     : ${ATTENTION_HEADS}"
echo " D_FF                : ${D_FF}"
echo " NUM_FFN_LAYERS      : ${NUM_FFN_LAYERS}"
echo " DROPOUT             : ${DROPOUT}"
echo " EPOCHS              : ${EPOCHS}"
echo " LR                  : ${LR}"
echo " WEIGHT_DECAY        : ${WEIGHT_DECAY}"
echo " MASKING_RATE        : ${MASKING_RATE}"
echo " MASK_AUGMENTATIONS  : ${MASK_AUGMENTATIONS}"
echo " MASKED_LOSS_WEIGHT  : ${MASKED_LOSS_WEIGHT}"
echo " OBSERVED_LOSS_WEIGHT: ${OBSERVED_LOSS_WEIGHT}"
echo " DEVICE              : ${DEVICE}"
echo " MAX_ITEM            : ${MAX_ITEM}"
echo " ITEM_DROPOUT_RATE   : ${ITEM_DROPOUT_RATE}"
echo " ANNOT_DROPOUT_RATE  : ${ANNOTATOR_DROPOUT_RATE}"
echo " TYPE_EMBEDDING_INIT : ${TYPE_EMBEDDING_INIT}"
echo " USE_POINTER         : ${USE_POINTER}"
echo " SCALE_SHARED_REL    : ${SCALE_SHARED_REL}"
echo " TRANSDUCTIVE        : $([ "${PROTOCOL}" = "transductive" ] && echo "true" || echo "false")"
echo " USE_EMBED_UNEMBED   : ${USE_EMBED_UNEMBED}"

for RAW_SIZE in "${SIZE_LIST[@]}"; do
    SIZE="$(echo "${RAW_SIZE}" | xargs)"
    RUN_START=$SECONDS

    if [[ "${AXIS}" == "item" ]]; then
        if [[ "${PROTOCOL}" == "transductive" ]]; then
            SPLIT_NAME="${SPLIT_PREFIX}_Item_T_${SIZE}"
            DATA_DIR="${DATA_ROOT}/ItemSplits/Transductive/${SPLIT_NAME}"
        else
            SPLIT_NAME="${SPLIT_PREFIX}_Item_NT_${SIZE}"
            DATA_DIR="${DATA_ROOT}/ItemSplits/NonTransductive/${SPLIT_NAME}"
        fi
    else
        if [[ "${PROTOCOL}" == "transductive" ]]; then
            SPLIT_NAME="${SPLIT_PREFIX}_Annot_T_${SIZE}"
            DATA_DIR="${DATA_ROOT}/AnnotSplits/Transductive/${SPLIT_NAME}"
        else
            SPLIT_NAME="${SPLIT_PREFIX}_Annot_NT_${SIZE}"
            DATA_DIR="${DATA_ROOT}/AnnotSplits/NonTransductive/${SPLIT_NAME}"
        fi
    fi

    RUN_NAME="${SPLIT_NAME}_MARFORMER"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

    echo ""
    echo "------------------------------------------------------------"
    echo " RUN_NAME  : ${RUN_NAME}"
    echo " DATA_DIR   : ${DATA_DIR}"
    echo " OUTPUT_DIR : ${RUN_DIR}"
    echo "------------------------------------------------------------"

    python -u -m imputer.entity_mf.train \
        --data-dir               "${DATA_DIR}"             \
        --run-name               "${RUN_NAME}"             \
        --output-root            "${OUTPUT_ROOT}"          \
        --seed                   "${SEED}"                 \
        --embedding-dim          "${EMBEDDING_DIM}"        \
        --num-layers             "${NUM_LAYERS}"           \
        --attention-heads        "${ATTENTION_HEADS}"      \
        --d-ff                   "${D_FF}"                 \
        --num-ffn-layers         "${NUM_FFN_LAYERS}"       \
        --dropout                "${DROPOUT}"              \
        --item-dropout-rate      "${ITEM_DROPOUT_RATE}"    \
        --annotator-dropout-rate "${ANNOTATOR_DROPOUT_RATE}" \
        --epochs                 "${EPOCHS}"               \
        --lr                     "${LR}"                   \
        --lr-schedule            "${LR_SCHEDULE}"          \
        --lr-min                 "${LR_MIN}"               \
        --weight-decay           "${WEIGHT_DECAY}"         \
        --masking-rate           "${MASKING_RATE}"         \
        --mask-augmentations     "${MASK_AUGMENTATIONS}"   \
        --masked-loss-weight     "${MASKED_LOSS_WEIGHT}"   \
        --observed-loss-weight   "${OBSERVED_LOSS_WEIGHT}" \
        --device                 "${DEVICE}"               \
        --max-item               "${MAX_ITEM}"             \
        --type-embedding-init    "${TYPE_EMBEDDING_INIT}"  \
        --item-reg-weight        "${ITEM_REG_WEIGHT}"      \
        --attribute-reg-weight   "${ATTRIBUTE_REG_WEIGHT}" \
        --annotator-reg-weight   "${ANNOTATOR_REG_WEIGHT}" \
        ${TRANSDUCTIVE_FLAG}                               \
        ${PER_HEAD_FLAG}                                   \
        ${SCALE_FLAG}                                      \
        ${POINTER_FLAG}                                    \
        ${REL_VALUE_FLAG}                                  \
        ${ADDONE_FLAG}                                     \
        ${DEVNORM_FLAG}                                    \
        ${GRAPHMASK_FLAG}                                  \
        ${LLM_DIST_FLAG}                                   \
        ${EMBED_UNEMBED_FLAG}                              \
        ${OVERWRITE_FLAG}

    RUN_ELAPSED=$(( SECONDS - RUN_START ))
    PAYLOAD_JSON=$(cat <<EOF
{
  "launcher_type": "marformer_train",
  "family": "${FAMILY_LABEL}",
  "axis": "${AXIS}",
  "protocol": "${PROTOCOL}",
  "size": ${SIZE},
  "data_dir": "${DATA_DIR}",
  "run_name": "${RUN_NAME}",
  "output_dir": "${RUN_DIR}",
  "elapsed_seconds": ${RUN_ELAPSED},
  "hyperparameters": {
    "seed": ${SEED},
    "embedding_dim": ${EMBEDDING_DIM},
    "num_layers": ${NUM_LAYERS},
    "attention_heads": ${ATTENTION_HEADS},
    "d_ff": ${D_FF},
    "num_ffn_layers": ${NUM_FFN_LAYERS},
    "dropout": ${DROPOUT},
    "epochs": ${EPOCHS},
    "lr": ${LR},
    "lr_schedule": "${LR_SCHEDULE}",
    "lr_min": ${LR_MIN},
    "weight_decay": ${WEIGHT_DECAY},
    "masking_rate": ${MASKING_RATE},
    "mask_augmentations": ${MASK_AUGMENTATIONS},
    "masked_loss_weight": ${MASKED_LOSS_WEIGHT},
    "observed_loss_weight": ${OBSERVED_LOSS_WEIGHT},
    "device": "${DEVICE}",
    "max_item": ${MAX_ITEM},
    "item_dropout_rate": ${ITEM_DROPOUT_RATE},
    "annotator_dropout_rate": ${ANNOTATOR_DROPOUT_RATE},
    "item_reg_weight": ${ITEM_REG_WEIGHT},
    "attribute_reg_weight": ${ATTRIBUTE_REG_WEIGHT},
    "annotator_reg_weight": ${ANNOTATOR_REG_WEIGHT},
    "type_embedding_init": "${TYPE_EMBEDDING_INIT}",
    "use_per_head_rel": ${USE_PER_HEAD_REL},
    "scale_shared_rel": ${SCALE_SHARED_REL},
    "use_pointer": ${USE_POINTER},
    "use_rel_value": ${USE_REL_VALUE},
    "use_addone_attn": ${USE_ADDONE_ATTN},
    "use_deviation_norm": ${USE_DEVIATION_NORM},
    "use_graph_mask": ${USE_GRAPH_MASK},
    "llm_input_dist": ${LLM_INPUT_DIST},
    "transductive_learning": $([ "${PROTOCOL}" = "transductive" ] && echo true || echo false),
    "use_embed_unembed": ${USE_EMBED_UNEMBED}
  }
}
EOF
)
    domain3_write_json_payload "${RUN_DIR}/launcher_metadata.json" "${PAYLOAD_JSON}"

    echo "  ↳ completed in $(( RUN_ELAPSED / 60 ))m $(( RUN_ELAPSED % 60 ))s"
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
domain3_print_block "${DATASET_LABEL} | ${FAMILY_LABEL} | ${AXIS} | ${PROTOCOL} complete"
echo " TOTAL_TIME  : $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo " OUTPUT_ROOT : ${OUTPUT_ROOT}"
