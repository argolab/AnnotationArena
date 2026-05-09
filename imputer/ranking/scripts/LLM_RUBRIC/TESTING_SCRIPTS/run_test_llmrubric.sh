#!/bin/bash

# LOCAL — run test evaluation (best + last checkpoint) on all LLM-Rubric runs.
# Usage: bash scripts/TESTING_SCRIPTS/run_test_llmrubric.sh

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

DEVICE="cuda"

SCRIPT_START=$SECONDS

# ── Run directories ───────────────────────────────────────────────────────────
RUN_DIRS=(
    # MARFORMER
    "RESULTS/MARFORMER/LLM_RUBRIC-2/LLMRubric_225_25_9_150"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_10"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_20"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_30"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_40"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_50"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_75"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_100"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_125"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_150"
    # "RESULTS/MARFORMER/LLM_RUBRIC/LLMRubric_225_25_9_175"
    # MARFORMER_HARD_MASK
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_10"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_20"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_30"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_40"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_50"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_75"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_100"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_125"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_150"
    # "RESULTS/MARFORMER_HARD_MASK/LLM_RUBRIC/LLMRubric_225_25_9_175"
)

echo ""
echo "============================================================"
echo " LLM-Rubric test evaluation — best + last checkpoints"
echo " Device: ${DEVICE}"
echo "============================================================"

for RUN_DIR in "${RUN_DIRS[@]}"; do
    if [ ! -d "$RUN_DIR" ]; then
        echo ""; echo "  [skip] ${RUN_DIR}  (not found)"; continue
    fi
    RUN_START=$SECONDS
    echo ""; echo "--- ${RUN_DIR} ---"; echo ""

    python -u -m imputer.entity_mf.test \
        --run-dir    "$RUN_DIR"  \
        --checkpoint both        \
        --device     "$DEVICE"

    RUN_ELAPSED=$(( SECONDS - RUN_START ))
    echo "  ↳ done in $(( RUN_ELAPSED / 60 ))m $(( RUN_ELAPSED % 60 ))s"
done

TOTAL_ELAPSED=$(( SECONDS - SCRIPT_START ))
echo ""
echo "============================================================"
echo " All done. Total time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo "============================================================"
