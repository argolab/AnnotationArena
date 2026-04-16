#!/usr/bin/env bash
# Submit the full grok ablation pool for one large puzzle (10k steps, 20% mask, live curves).
#
# Usage (from imputer/ranking):
#   ./scripts/test_scripts/grok_success_ablation/submit_grok_ablations_puzzle_s10000.sh 4x4_r1
#   ./scripts/test_scripts/grok_success_ablation/submit_grok_ablations_puzzle_s10000.sh 6x6_r1
#   ./scripts/test_scripts/grok_success_ablation/submit_grok_ablations_puzzle_s10000.sh 5x5_r2
#   ./scripts/test_scripts/grok_success_ablation/submit_grok_ablations_puzzle_s10000.sh 7x7_r2
#
# Env overrides: TIME, PARTITION, GROK_LIVE_CURVES_EVERY, PREVIEW=1, etc.

set -euo pipefail

PUZZLE="${1:-}"
case "${PUZZLE}" in
  4x4_r1)
    export GROK_N=4 GROK_M=4 GROK_D=1
    export GROK_OUT_ROOT="${GROK_OUT_ROOT:-OUTPUT/grok_ablation_4x4_r1_s10000}"
    export GROK_JOB_PREFIX="${GROK_JOB_PREFIX:-g4x4r1_}"
    ;;
  6x6_r1)
    export GROK_N=6 GROK_M=6 GROK_D=1
    export GROK_OUT_ROOT="${GROK_OUT_ROOT:-OUTPUT/grok_ablation_6x6_r1_s10000}"
    export GROK_JOB_PREFIX="${GROK_JOB_PREFIX:-g6x6r1_}"
    ;;
  5x5_r2)
    export GROK_N=5 GROK_M=5 GROK_D=2
    export GROK_OUT_ROOT="${GROK_OUT_ROOT:-OUTPUT/grok_ablation_5x5_r2_s10000}"
    export GROK_JOB_PREFIX="${GROK_JOB_PREFIX:-g5x5r2_}"
    ;;
  7x7_r2)
    export GROK_N=7 GROK_M=7 GROK_D=2
    export GROK_OUT_ROOT="${GROK_OUT_ROOT:-OUTPUT/grok_ablation_7x7_r2_s10000}"
    export GROK_JOB_PREFIX="${GROK_JOB_PREFIX:-g7x7r2_}"
    ;;
  *)
    echo "Usage: $0 {4x4_r1|6x6_r1|5x5_r2|7x7_r2}" >&2
    exit 2
    ;;
esac

export TIME="${TIME:-36:00:00}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/submit_grok_success_ablations.sh"
