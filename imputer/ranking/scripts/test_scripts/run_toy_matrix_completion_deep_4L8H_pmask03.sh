#!/bin/bash
# Toy matrix completion — deep/wide run (4 layers, 8 heads).
# EntityMarformer flags mostly mirror run_train.sh; toy defaults use_rel_value=True (add --no-rel-value for STAN parity).
#   kaiming init, dropout 0.1, weight decay 0.01, shared relational bias + scale, pointer on,
#   no per-head-rel / addone / deviation-norm / graph-mask / learned-emb.
# (Toy graphs use mc_entry only; pointer K_aug is inactive but Q_ptr still matches production wiring.)

set -e

PYTHONPATH=. python toy_scripts/toy_matrix_completion.py \
  --steps 50000 \
  --num-train-graphs 256 \
  --num-test-graphs 20 \
  --N 7 \
  --M 7 \
  --D 2 \
  --mask-rate 0.3 \
  --embedding-dim 80 \
  --num-layers 4 \
  --attn-heads 8 \
  --d-ff 128 \
  --num-ffn-layers 1 \
  --dropout 0.1 \
  --type-embedding-init kaiming \
  --weight-decay 0.01 \
  --resample-train-each-step \
  --out-dir OUTPUT/toy_matrix_completion_curves_deep_4L8H_pmask03
