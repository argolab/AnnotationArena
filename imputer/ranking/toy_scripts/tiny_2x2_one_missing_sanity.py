#!/usr/bin/env python3
from __future__ import annotations

"""
Sanity check: low-rank matrix completion with exactly ONE fixed masked cell.

Motivation:
  With N=M=2 and rank=1, y = u v^T is fully determined by 3 observed entries if the masked
  coordinate is fixed. This is a near-trivial memorization / "few rules" probe.

This script runs two baselines back-to-back:
  - bert_aligned_mc_only.py (vanilla TransformerEncoder)
  - entity_mf_mc_only.py (EntityMarformer path, mc_entry-only, no relations)

Defaults are intentionally small so you can iterate quickly; scale up with --steps.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_module(*, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Required for dataclasses / typing introspection when loading from a file path.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description="Low-rank masking sanity runner (BERT vs EntityMF).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--M", type=int, default=2)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--latent-sample-min", type=float, default=-1.0)
    p.add_argument("--latent-sample-max", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Stdout log interval for BERT and EntityMF (EntityMF: 0 disables periodic logs). Default: same as --eval-every.",
    )
    p.add_argument("--live-curves-every", type=int, default=100)
    p.add_argument(
        "--mask-mode",
        type=str,
        default="fixed",
        choices=("fixed", "random_one", "random"),
        help=(
            "fixed: one fixed missing coordinate; "
            "random_one: exactly one uniformly random missing coordinate per sample; "
            "random: independent MCAR masking using --mask-prob."
        ),
    )
    p.add_argument("--mask-prob", type=float, default=0.3, help="Used by --mask-mode random (MCAR mask rate).")
    p.add_argument("--fixed-mask-i", type=int, default=1)
    p.add_argument("--fixed-mask-j", type=int, default=1)
    p.add_argument(
        "--sweep-fixed-masks",
        action="store_true",
        help="Run all fixed one-missing coordinates for N x M. Writes subdirs mask_i_j/.",
    )
    p.add_argument(
        "--emf-only",
        action="store_true",
        help="Skip the BERT-aligned baseline and run only the EntityMF (mc_entry-only) branch.",
    )
    p.add_argument("--out-dir", type=str, required=True)

    # Model sizes (keep small by default)
    p.add_argument("--bert-hidden-size", type=int, default=64)
    p.add_argument("--bert-num-layers", type=int, default=2)
    p.add_argument("--bert-num-heads", type=int, default=2)
    p.add_argument("--bert-intermediate-size", type=int, default=128)
    p.add_argument("--bert-positional-scheme", type=str, default="rowcol_concat", choices=("flat", "rowcol_concat"))

    p.add_argument("--emf-embedding-dim", type=int, default=64)
    p.add_argument("--emf-num-layers", type=int, default=2)
    p.add_argument("--emf-attn-heads", type=int, default=2)
    p.add_argument("--emf-d-ff", type=int, default=128)
    p.add_argument("--emf-num-ffn-layers", type=int, default=1)
    p.add_argument("--emf-type-embedding-init", type=str, default="kaiming", choices=("normal", "scaled_normal", "kaiming"))

    args = p.parse_args()

    log_n = args.eval_every if args.log_every is None else args.log_every
    bert_log_every = max(1, log_n) if log_n > 0 else 10**9
    emf_log_every = max(0, log_n)

    root = Path(__file__).resolve().parent
    bert_mod = _load_module(name="bert_aligned_mc_only", path=root / "bert_aligned_mc_only.py")
    emf_mod = _load_module(name="entity_mf_mc_only", path=root / "entity_mf_mc_only.py")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mask_mode == "fixed" and args.sweep_fixed_masks:
        masks = [(i, j) for i in range(args.N) for j in range(args.M)]
    else:
        masks = [(args.fixed_mask_i, args.fixed_mask_j)]
    if args.mask_mode == "random" and not (0.0 < args.mask_prob < 1.0):
        raise ValueError(f"--mask-prob must be in (0,1) for --mask-mode random, got {args.mask_prob}")

    # Print resolved run configuration once up front so logs are self-describing.
    print(
        "Run config: "
        f"N={args.N} M={args.M} rank={args.rank} steps={args.steps} "
        f"train_batch_size={args.train_batch_size} eval_batch_size={args.eval_batch_size} "
        f"mask_mode={args.mask_mode} mask_prob={args.mask_prob} "
        f"eval_every={args.eval_every} log_every={log_n} live_curves_every={args.live_curves_every} "
        f"emf_only={args.emf_only}"
    )

    for mi, mj in masks:
        sub = out / f"mask_{mi}_{mj}" if args.sweep_fixed_masks else out
        sub.mkdir(parents=True, exist_ok=True)

        common = dict(
            seed=args.seed,
            steps=args.steps,
            n_rows=args.N,
            n_cols=args.M,
            rank=args.rank,
            mask_prob=args.mask_prob,
            mask_mode=args.mask_mode,
            fixed_masked_i=mi,
            fixed_masked_j=mj,
            latent_sample_min=args.latent_sample_min,
            latent_sample_max=args.latent_sample_max,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            live_curves_every=args.live_curves_every,
        )

        if not args.emf_only:
            print("Running BERT-aligned baseline...")
            bert_mod.run_experiment(
                out_dir=sub / "bert_aligned_mc_only",
                positional_scheme=args.bert_positional_scheme,
                hidden_size=args.bert_hidden_size,
                num_layers=args.bert_num_layers,
                num_heads=args.bert_num_heads,
                intermediate_size=args.bert_intermediate_size,
                dropout=0.0,
                log_every=bert_log_every,
                **common,
            )

        print("Running EntityMF...")
        emf_mod.run(
            out_dir=sub / "entity_mf_mc_only",
            embedding_dim=args.emf_embedding_dim,
            num_layers=args.emf_num_layers,
            attn_heads=args.emf_attn_heads,
            d_ff=args.emf_d_ff,
            num_ffn_layers=args.emf_num_ffn_layers,
            dropout=0.0,
            type_embedding_init=args.emf_type_embedding_init,
            use_mc_mask_bit=True,
            log_every=emf_log_every,
            **common,
        )

        if args.mask_mode == "fixed":
            label = f"mask=({mi},{mj})"
        elif args.mask_mode == "random_one":
            label = "mask=random_one"
        else:
            label = f"mask=random(p={args.mask_prob})"
        parts = [f"Done {label}. Wrote:"]
        if not args.emf_only:
            parts.append(f"  {(sub / 'bert_aligned_mc_only').as_posix()}")
        parts.append(f"  {(sub / 'entity_mf_mc_only').as_posix()}")
        print("\n".join(parts))


if __name__ == "__main__":
    main()
