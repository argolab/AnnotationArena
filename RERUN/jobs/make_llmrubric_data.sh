#!/bin/bash
# Build the LLM-Rubric bundles for one run, from the raw TSVs.
#
#   ./jobs/make_llmrubric_data.sh --run-id baseline
#
# Runs in seconds on a login node -- no sbatch needed.
#
# make_llmrubric_splits.py hardcodes its output to imputer/ranking/DATA/LLM_RUBRIC
# (RANKING_ROOT is Path(__file__).parents[2], no CLI override). This wrapper
# writes there and then relocates the result into the run's own DATA tree, so
# imputer/ranking/DATA stays clear -- make_figures.sh refuses to bind over a
# real directory there, by design.
set -euo pipefail

source "$(dirname "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")")/config/cluster.sh"

RUN_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2 ;;
        -h|--help) echo "usage: $0 --run-id ID"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -n "${RUN_ID}" ]] || { echo "--run-id is required" >&2; exit 1; }

rerun_activate_env

STAGING="${RANKING_ROOT}/DATA"
DEST="$(rerun_run_dir "${RUN_ID}")/DATA/LLM_RUBRIC"

if [[ -e "${STAGING}" && ! -L "${STAGING}" ]] && [[ -n "$(ls -A "${STAGING}" 2>/dev/null)" ]]; then
    echo "REFUSING: ${STAGING} already exists and is non-empty." >&2
    echo "Move it aside first; this script needs that path as scratch." >&2
    exit 1
fi

cd "${RANKING_ROOT}"
rerun_run python scripts/convert-llm-rubric/make_llmrubric_splits.py

if [[ -n "${RERUN_DRY:-}" ]]; then exit 0; fi

mkdir -p "$(dirname "${DEST}")"
rm -rf "${DEST}"
mv "${STAGING}/LLM_RUBRIC" "${DEST}"
rmdir "${STAGING}" 2>/dev/null || true

echo
echo "LLM-Rubric bundles -> ${DEST}"
ls "${DEST}"
