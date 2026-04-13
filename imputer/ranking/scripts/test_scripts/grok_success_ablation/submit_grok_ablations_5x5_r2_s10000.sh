#!/usr/bin/env bash
# 5×5 rank-2 matrix (D=2), 20% mask, 10k steps — full ablation pool.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_grok_ablations_puzzle_s10000.sh" 5x5_r2
