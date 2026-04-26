#!/usr/bin/env python3
"""
run_baselines.py
----------------
Train and evaluate ReMasker or MIWAE on an annotation imputation bundle.

Usage
-----
    # ReMasker on LLMRubric split
    python BASELINES/run_baselines.py \
        --method remasker \
        --data-bundle DATA/LLM_RUBRIC/LLMRubric_225_25_9_10/data_bundle.json \
        --output-dir  RESULTS/BASELINES/LLMRubric_225_25_9_10/remasker

    # MIWAE on SummEval split
    python BASELINES/run_baselines.py \
        --method miwae \
        --data-bundle DATA/SUMMEVAL/SummEval_1600_8_4_50/data_bundle.json \
        --output-dir  RESULTS/BASELINES/SummEval_1600_8_4_50/miwae \
        --epochs 500

The script:
  1. Loads the data bundle (train / test split already encoded in the bundle).
  2. Trains the model on K_train items using all observed annotations (including
     always-observed annotators such as the LLM in LLMRubric or turkers in SummEval).
  3. At test time: passes only the context (always-observed) annotations, and
     evaluates predictions for the target (missing) annotations.
  4. Saves: model checkpoint, per-entry prediction probabilities, summary metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running from the repo root or from BASELINES/
sys.path.insert(0, str(Path(__file__).parent))
from bundle_to_matrix import load_bundle, compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Run ReMasker or MIWAE baseline on annotation bundle")

    # Required
    p.add_argument("--method",      required=True, choices=["remasker", "miwae"],
                   help="Which baseline to run")
    p.add_argument("--data-bundle", required=True,
                   help="Path to data_bundle.json")
    p.add_argument("--output-dir",  required=True,
                   help="Directory to write model checkpoint and results")

    # Training
    p.add_argument("--epochs",      type=int,   default=None,
                   help="Training epochs (default: 300 for remasker, 500 for miwae)")
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cpu",
                   help="'cpu', 'cuda', or 'cuda:0' etc.")
    p.add_argument("--train-mask-ratio", type=float, default=0.5,
                   help="Random observed-cell masking ratio used only for transductive item bundles.")

    # ReMasker-specific
    p.add_argument("--embed-dim",       type=int,   default=32)
    p.add_argument("--depth",           type=int,   default=4)
    p.add_argument("--num-heads",       type=int,   default=4)
    p.add_argument("--decoder-depth",   type=int,   default=2)
    p.add_argument("--mask-ratio",      type=float, default=0.5)
    p.add_argument("--weight-decay",    type=float, default=0.05)

    # MIWAE-specific
    p.add_argument("--latent-dim",  type=int,   default=4,
                   help="Latent dimension d (MIWAE)")
    p.add_argument("--hidden-dim",  type=int,   default=128,
                   help="MLP hidden units h (MIWAE)")
    p.add_argument("--K",           type=int,   default=20,
                   help="IS samples during training (MIWAE)")
    p.add_argument("--L",           type=int,   default=50,
                   help="IS samples during imputation (MIWAE)")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Method     : {args.method.upper()}")
    print(f" Bundle     : {args.data_bundle}")
    print(f" Output dir : {out_dir}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    data = load_bundle(args.data_bundle)
    C    = data.meta["C"]

    print(f"Data dimensions:")
    print(f"  K_train = {data.meta['K_train']},  K_test = {data.meta['K_test']}")
    print(f"  J = {data.meta['J']},  I = {data.meta['I']},  C = {C},  D = {data.meta['D']}")
    print(f"  Training mode: {data.training_mode}")
    print(f"  Fit rows: {data.X_fit.shape[0]}")
    print(f"  Context annotators (observed at test): {sorted(data.context_annotators)}")
    print(f"  Target  annotators (to predict):       {sorted(data.target_annotators)}")
    print(f"  Test targets to evaluate: {len(data.test_targets)}\n")

    if data.X_fit.shape[0] == 0:
        raise ValueError("Training matrix is empty. No observed rows available for baseline fitting.")

    # ── Build and train model ─────────────────────────────────────────────────
    t0 = time.time()

    if args.method == "remasker":
        from remasker import CategoricalReMasker
        epochs = args.epochs if args.epochs is not None else 300
        model = CategoricalReMasker(
            num_classes=C,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            decoder_embed_dim=args.embed_dim,
            decoder_depth=args.decoder_depth,
            decoder_num_heads=args.num_heads,
            mask_ratio=args.mask_ratio,
            max_epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            seed=args.seed,
        )

    elif args.method == "miwae":
        from miwae import CategoricalMIWAE
        epochs = args.epochs if args.epochs is not None else 500
        model = CategoricalMIWAE(
            num_classes=C,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            K=args.K,
            L=args.L,
            max_epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )

    print(f"Training {args.method.upper()} for {epochs} epochs...")
    fit_kwargs = {}
    if data.training_mode == "transductive_item_domain3":
        fit_kwargs["random_mask_observed"] = True
        fit_kwargs["train_mask_ratio"] = args.train_mask_ratio
    else:
        fit_kwargs["context_cols_mask"] = data.context_cols_mask
        fit_kwargs["target_cols_mask"] = data.target_cols_mask

    model.fit(data.X_fit, data.M_fit, **fit_kwargs)
    train_elapsed = time.time() - t0
    print(f"\nTraining complete in {train_elapsed:.1f}s\n")

    # ── Save model checkpoint ─────────────────────────────────────────────────
    ckpt_path = str(out_dir / f"{args.method}_model.pt")
    model.save(ckpt_path)

    # ── Inference on test items ───────────────────────────────────────────────
    print("Running inference on test items...")
    t1 = time.time()
    probs = model.transform(data.X_test, data.M_test)   # (K_test, D, C)
    inf_elapsed = time.time() - t1
    print(f"Inference complete in {inf_elapsed:.1f}s\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(probs, data.test_targets)
    print(f"=== TEST RESULTS ===")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Mean NLL  : {metrics['mean_nll']:.4f}")
    print(f"  N targets : {metrics['n']}")
    print()

    # ── Save predictions and metrics ──────────────────────────────────────────
    # Per-entry predictions
    pred_entries = []
    for (row, col, label) in data.test_targets:
        p_vec = probs[row, col, :].tolist()
        pred_entries.append({
            "test_item_row": row,
            "col_idx":       col,
            "true_label":    label,
            "pred_label":    int(np.argmax(p_vec)),
            "probs":         p_vec,
        })

    preds_path = out_dir / "test_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(pred_entries, f)
    print(f"Per-entry predictions saved → {preds_path}")

    # Summary metrics + config
    summary = {
        "method":  args.method,
        "bundle":  str(args.data_bundle),
        "metrics": metrics,
        "train_elapsed_s": round(train_elapsed, 2),
        "inf_elapsed_s":   round(inf_elapsed, 2),
        "config": vars(args),
        "data_meta": data.meta,
        "context_annotators": sorted(data.context_annotators),
        "target_annotators":  sorted(data.target_annotators),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")

    print(f"\n{'='*60}")
    print(f" Done.  accuracy={metrics['accuracy']:.4f}  nll={metrics['mean_nll']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
