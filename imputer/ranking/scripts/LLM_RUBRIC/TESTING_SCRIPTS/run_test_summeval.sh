#!/bin/bash

# LOCAL — run test evaluation (best + last checkpoint) on all SummEval runs.
# Usage: bash scripts/TESTING_SCRIPTS/run_test_summeval.sh

cd /Users/prabhavsingh/Documents/JHU/JHUResearch/EntityMarformer/imputer/ranking
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

DEVICE="cuda"

SCRIPT_START=$SECONDS

# ── Run directories ───────────────────────────────────────────────────────────
RUN_DIRS=(
    # MARFORMER
    "RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_50"
    "RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_100"
    "RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_500"
    "RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_750"
    "RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_1000"
    "RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_1280"
    # MARFORMER_HARD_MASK
    "RESULTS/MARFORMER_HARD_MASK/SUMMEVAL/SummEval_1600_8_4_50"
    "RESULTS/MARFORMER_HARD_MASK/SUMMEVAL/SummEval_1600_8_4_100"
    "RESULTS/MARFORMER_HARD_MASK/SUMMEVAL/SummEval_1600_8_4_500"
    "RESULTS/MARFORMER_HARD_MASK/SUMMEVAL/SummEval_1600_8_4_750"
    "RESULTS/MARFORMER_HARD_MASK/SUMMEVAL/SummEval_1600_8_4_1000"
    "RESULTS/MARFORMER_HARD_MASK/SUMMEVAL/SummEval_1600_8_4_1280"
)

echo ""
echo "============================================================"
echo " SummEval test evaluation — best + last checkpoints"
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
