#!/usr/bin/env bash
# 4×4 rank-1 matrix, 20% mask, 10k steps — full ablation pool.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_grok_ablations_puzzle_s10000.sh" 4x4_r1
