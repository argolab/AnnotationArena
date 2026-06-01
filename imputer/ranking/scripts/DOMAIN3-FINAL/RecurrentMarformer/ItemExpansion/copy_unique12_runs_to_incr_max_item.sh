#!/bin/bash
# Copy all DOMAIN3-OLD-UNIQUE12 run dirs (checkpoints through ~600 epochs) into a
# sibling results root for max_item=200 continuation. Does not modify originals.
#
#   cd imputer/ranking
#   bash scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/copy_unique12_runs_to_incr_max_item.sh
#
# Env: SRC_ROOT, DST_ROOT (defaults below). Skips copy if destination run dir exists.

set -euo pipefail

RANKING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${RANKING_ROOT}"

SRC_ROOT="${SRC_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12}"
DST_ROOT="${DST_ROOT:-RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE12_INCR_MAX_ITEM}"

if [ ! -d "$SRC_ROOT" ]; then
    echo "ERROR: source root not found: $SRC_ROOT" >&2
    exit 1
fi

mkdir -p "$DST_ROOT"

shopt -s nullglob
RUN_DIRS=( "$SRC_ROOT"/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_* )
shopt -u nullglob

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: no run dirs under $SRC_ROOT" >&2
    exit 1
fi

echo "============================================================"
echo " Copy UNIQUE12 runs -> INCR_MAX_ITEM"
echo " SRC: ${SRC_ROOT}"
echo " DST: ${DST_ROOT}"
echo " Runs: ${#RUN_DIRS[@]}"
echo "============================================================"

for src in "${RUN_DIRS[@]}"; do
    name="$(basename "$src")"
    dest="${DST_ROOT}/${name}"
    if [ -d "$dest" ]; then
        echo "SKIP (exists): $name"
        continue
    fi
    echo "cp -a $name"
    cp -a "$src" "$dest"
done

echo ""
echo "Done. Copied runs are under: ${DST_ROOT}"
