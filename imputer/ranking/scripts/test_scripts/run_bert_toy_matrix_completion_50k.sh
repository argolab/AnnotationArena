#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHONPATH=. python toy_scripts/bert_toy_matrix_completion.py \
  --steps 50000 \
  --live-curves-every 100 \
  --out-dir OUTPUT/bert_toy_matrix_completion_50k_live100

