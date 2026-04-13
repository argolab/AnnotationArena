#!/usr/bin/env bash
# Shared CLI for the "grok success" 2×2 matrix-completion setup.
#
# Your manual command + imputed toy defaults (toy_matrix_completion.py):
#
#   You passed:
#     --num-train-graphs 10 --num-test-graphs 10 --steps 2000 --seed 0
#     --N 2 --M 2 --D 1  --mask-rate 0.2
#     --dropout 0 --weight-decay 0  --resample-train-each-step
#     --show-correct-vector --multiplication-head
#
#   Defaults imputed (when omitted from CLI):
#     --latent-sample-std 3.0
#     --embedding-dim 32  --num-layers 4  --attn-heads 4  --d-ff 128  --num-ffn-layers 1
#     --lr 3e-4
#     --type-embedding-init kaiming
#     --entry-input-tag-scale 1e-3
#
# Override any knob by exporting before calling grok_run, e.g. GROK_LAYERS=3 GROK_EMB=24
# Large sweeps: export GROK_OUT_ROOT=OUTPUT/grok_ablation_6x6_r1_s10000 (and N,M,D,STEPS) before
# the worker script; per-run outputs go under that directory.
#
# shellcheck shell=bash
set -euo pipefail

# grok_success_ablation/ -> test_scripts/ -> scripts/ -> imputer/ranking (where toy_scripts/ lives)
_GROK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${_GROK_ROOT}"

: "${GROK_OUT_ROOT:=OUTPUT/grok_success_ablation}"
: "${GROK_LIVE_CURVES_EVERY:=100}"

: "${GROK_STEPS:=2000}"
: "${GROK_SEED:=0}"
: "${GROK_N:=2}"
: "${GROK_M:=2}"
: "${GROK_D:=1}"
: "${GROK_MASK_RATE:=0.2}"
: "${GROK_LATENT_SAMPLE_STD:=3.0}"
: "${GROK_EMB:=32}"
: "${GROK_LAYERS:=4}"
: "${GROK_HEADS:=4}"
: "${GROK_D_FF:=128}"
: "${GROK_FFN_LAYERS:=1}"
: "${GROK_DROPOUT:=0}"
: "${GROK_LR:=3e-4}"
: "${GROK_WEIGHT_DECAY:=0}"
: "${GROK_TRAIN_GRAPHS:=10}"
: "${GROK_TEST_GRAPHS:=10}"
: "${GROK_TYPE_INIT:=kaiming}"
: "${GROK_ENTRY_TAG:=1e-3}"

grok_run() {
  local out_dir=$1
  shift
  PYTHONPATH=. python toy_scripts/toy_matrix_completion.py \
    --num-train-graphs "${GROK_TRAIN_GRAPHS}" \
    --num-test-graphs "${GROK_TEST_GRAPHS}" \
    --steps "${GROK_STEPS}" \
    --seed "${GROK_SEED}" \
    --N "${GROK_N}" \
    --M "${GROK_M}" \
    --D "${GROK_D}" \
    --mask-rate "${GROK_MASK_RATE}" \
    --latent-sample-std "${GROK_LATENT_SAMPLE_STD}" \
    --embedding-dim "${GROK_EMB}" \
    --num-layers "${GROK_LAYERS}" \
    --attn-heads "${GROK_HEADS}" \
    --d-ff "${GROK_D_FF}" \
    --num-ffn-layers "${GROK_FFN_LAYERS}" \
    --dropout "${GROK_DROPOUT}" \
    --lr "${GROK_LR}" \
    --weight-decay "${GROK_WEIGHT_DECAY}" \
    --type-embedding-init "${GROK_TYPE_INIT}" \
    --resample-train-each-step \
    --entry-input-tag-scale "${GROK_ENTRY_TAG}" \
    --live-curves-every "${GROK_LIVE_CURVES_EVERY}" \
    "$@" \
    --out-dir "${out_dir}"
}
