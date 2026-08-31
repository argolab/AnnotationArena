#!/bin/bash
#SBATCH --job-name=fig6-gen
#SBATCH --partition=med
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --comment=accept_cost
set -euo pipefail

# Re-derive the train/test subsplits from replicate-01's EXISTING base bundle,
# writing into a fresh data root. The base is never regenerated and
# replicate-01 is never written to, so the figure's existing dots cannot move.
#
# The full size lists are re-derived, not just the new ones, so the overlap can
# be checksummed against replicate-01 to prove the subsplitter is deterministic
# before any of the new bundles are trusted.

RERUN_DIR="${RERUN_DIR:-/weka/scratch/jhu/jeisner1/xwang397/Marformer/RERUN}"
source "${RERUN_DIR}/config/cluster.sh"
rerun_activate_env

SRC_BASE="${RERUN_DIR}/runs/replicate-01/DATA/STAN/DOMAIN3-SIGMOID/Base/DOMAIN3-SIGMOID_BASE_5000_50_9"
OUT_ROOT="${RERUN_DIR}/runs/fig6-01/DATA/STAN/DOMAIN3-SIGMOID"
mkdir -p "${OUT_ROOT}"

# Unchanged from replicate-01.
ITEM_TRANS_SIZES="100,200,300,400,500,600,700,800,900,1000,2000,3000,4000"
ANNOT_TRANS_SIZES="10,15,20,25,30,35,40,45,50"
# Extended: item NT gains 1000,2000,3000,4000 so the NT curve spans the same
# x-range as the transductive one (ceiling is 4900 = 5000 items - the 100-item
# corner).
ITEM_NT_SIZES="100,200,300,400,500,600,700,800,900,1000,2000,3000,4000"
# Annotator NT is NOT extended. The subsplitter enforces
# train_size <= total_annots - test_annotators, and the base has J=50 with 10
# reserved for the corner, so 40 is a hard ceiling. Annot_T_50 and Annot_NT_40
# both train on 40 annotators; the paper's NT curve stopping at 40 is a property
# of the data, not an omission. Extending it would need a new base with J>50.
ANNOT_NT_SIZES="10,15,20,25,30,35,40"

cd "${RANKING_ROOT}"
rerun_run python STAN/stan_code/scripts/domain3subsplit.py \
    --input-dir         "${SRC_BASE}"                  \
    --output-root       "${OUT_ROOT}"                  \
    --run-prefix        "DOMAIN3-SIGMOID"              \
    --item-trans-sizes  "${ITEM_TRANS_SIZES}"          \
    --item-nt-sizes     "${ITEM_NT_SIZES}"             \
    --annot-trans-sizes "${ANNOT_TRANS_SIZES}"         \
    --annot-nt-sizes    "${ANNOT_NT_SIZES}"            \
    --test-items        100                            \
    --test-annotators   10

echo "subsplit complete -> ${OUT_ROOT}"
