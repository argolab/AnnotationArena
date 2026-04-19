#!/usr/bin/env python3
from __future__ import annotations

"""
Matched clipped matrix-completion runner for:
  - BERT-aligned model (structured positional embedding + real-valued tokenizer)
  - EntityMF row/col-token model
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
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description="Run clipped matrix completion on BERT and EntityMF with matched settings.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--mask-mode", type=str, default="random", choices=("random", "fixed", "random_one"))
    p.add_argument("--mask-prob", type=float, default=0.3)
    p.add_argument("--fixed-mask-i", type=int, default=None)
    p.add_argument("--fixed-mask-j", type=int, default=None)
    p.add_argument("--clip-min", type=float, required=True)
    p.add_argument("--clip-max", type=float, required=True)
    p.add_argument("--latent-sample-min", type=float, default=-1.0)
    p.add_argument("--latent-sample-max", type=float, default=1.0)
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--live-curves-every", type=int, default=100)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--emf-only", action="store_true")

    # BERT
    p.add_argument("--bert-hidden-size", type=int, default=256)
    p.add_argument("--bert-num-layers", type=int, default=4)
    p.add_argument("--bert-num-heads", type=int, default=4)
    p.add_argument("--bert-intermediate-size", type=int, default=1024)
    p.add_argument("--bert-positional-scheme", type=str, default="rowcol_concat", choices=("flat", "rowcol_concat"))

    # EntityMF
    p.add_argument("--emf-embedding-dim", type=int, default=256)
    p.add_argument("--emf-num-layers", type=int, default=4)
    p.add_argument("--emf-attn-heads", type=int, default=4)
    p.add_argument("--emf-d-ff", type=int, default=1024)
    p.add_argument("--emf-num-ffn-layers", type=int, default=1)
    p.add_argument("--emf-type-embedding-init", type=str, default="kaiming", choices=("normal", "scaled_normal", "kaiming"))

    args = p.parse_args()

    if args.clip_max <= args.clip_min:
        raise ValueError(f"clip_max must be > clip_min, got [{args.clip_min}, {args.clip_max}]")

    root = Path(__file__).resolve().parent
    bert_mod = _load_module(name="bert_aligned_mc_only", path=root / "bert_aligned_mc_only.py")
    emf_mod = _load_module(name="entity_mf_rowcol_tokens", path=root / "entity_mf_rowcol_tokens.py")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(
        "Run config: "
        f"N={args.N} M={args.M} rank={args.rank} steps={args.steps} "
        f"mask_mode={args.mask_mode} mask_prob={args.mask_prob} "
        f"clip=[{args.clip_min},{args.clip_max}] "
        f"train_batch_size={args.train_batch_size} eval_batch_size={args.eval_batch_size}"
    )

    common = dict(
        seed=args.seed,
        steps=args.steps,
        n_rows=args.N,
        n_cols=args.M,
        rank=args.rank,
        mask_prob=args.mask_prob,
        mask_mode=args.mask_mode,
        fixed_masked_i=args.fixed_mask_i,
        fixed_masked_j=args.fixed_mask_j,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        latent_sample_min=args.latent_sample_min,
        latent_sample_max=args.latent_sample_max,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        eval_every=args.eval_every,
        live_curves_every=args.live_curves_every,
    )

    if not args.emf_only:
        print("Running BERT clipped baseline...")
        bert_mod.run_experiment(
            out_dir=out / "bert_aligned_clipped",
            hidden_size=args.bert_hidden_size,
            num_layers=args.bert_num_layers,
            num_heads=args.bert_num_heads,
            intermediate_size=args.bert_intermediate_size,
            positional_scheme=args.bert_positional_scheme,
            dropout=0.0,
            **common,
        )

    print("Running EntityMF clipped model...")
    emf_mod.run(
        out_dir=out / "entity_mf_rowcol_tokens_clipped",
        embedding_dim=args.emf_embedding_dim,
        num_layers=args.emf_num_layers,
        attn_heads=args.emf_attn_heads,
        d_ff=args.emf_d_ff,
        num_ffn_layers=args.emf_num_ffn_layers,
        dropout=0.0,
        type_embedding_init=args.emf_type_embedding_init,
        use_mc_mask_bit=True,
        **common,
    )


if __name__ == "__main__":
    main()

