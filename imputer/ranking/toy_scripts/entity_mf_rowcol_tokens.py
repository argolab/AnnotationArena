#!/usr/bin/env python3
from __future__ import annotations

"""
EntityMF variant with explicit row/col entity tokens (no fixed one-hot positional tail).

Compared to entity_mf_mc_only.py:
  - REMOVE fixed_feature = one_hot(row)+one_hot(col) on mc_entry tokens
  - ADD row and col entity tokens with variation enabled
  - ADD entry<->row and entry<->col relations/edges
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.data import EntityGraph, Relationship, Token
from imputer.entity_mf.model import EntityMarformer
from imputer.entity_mf.synthetic.types import RegressionSlices, SyntheticRegressionType
from imputer.entity_mf.types import VariationConfig


SEED = 42
NUM_STEPS = 50_000

N_ROWS = 4
N_COLS = 4
RANK = 1
MASK_PROB = 0.3

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 64

LATENT_SAMPLE_MIN = -1.0
LATENT_SAMPLE_MAX = 1.0

EMBEDDING_DIM = 512
NUM_LAYERS = 6
ATTN_HEADS = 4
D_FF = 2048
NUM_FFN_LAYERS = 1
DROPOUT = 0.0

LR = 3e-4
WEIGHT_DECAY = 0.0

TYPE_EMBEDDING_INIT = "kaiming"

EVAL_EVERY = 50
LIVE_CURVES_EVERY = 100
LOG_EVERY = 0

OUT_DIR = Path("OUTPUT/entity_mf_rowcol_tokens")
_LOG_Y_FLOOR = 1e-12


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _product_bounds(lo: float, hi: float) -> tuple[float, float]:
    cands = [lo * lo, lo * hi, hi * lo, hi * hi]
    return float(min(cands)), float(max(cands))


def _quantize_to_bins(x: torch.Tensor, *, num_bins: int, bin_min: float, bin_max: float) -> torch.Tensor:
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    if bin_max <= bin_min:
        raise ValueError(f"bin_max must be > bin_min, got [{bin_min}, {bin_max}]")
    x_clip = x.clamp(min=bin_min, max=bin_max)
    edges = torch.linspace(bin_min, bin_max, steps=num_bins + 1, device=x.device, dtype=x.dtype)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = torch.bucketize(x_clip, edges[1:-1], right=False)
    return centers[idx]


@dataclass
class Sample:
    graph: EntityGraph
    entry_targets: torch.Tensor  # [L_entry]
    entry_token_indices: List[int]  # absolute token indices in graph
    masked_entry_positions: List[int]  # positions in [0..L_entry)


def _build_types(*, n_rows: int, n_cols: int, use_mask_bit: bool = True) -> Dict[str, Any]:
    in_dim = 2 if use_mask_bit else 1
    return {
        "mc_row": SyntheticRegressionType(
            name="mc_row",
            slices=RegressionSlices(input_dim=0, output_dim=0),
            has_target=False,
            variation=VariationConfig(enabled=True, num_entities=n_rows, reg_weight=0.0),
        ),
        "mc_col": SyntheticRegressionType(
            name="mc_col",
            slices=RegressionSlices(input_dim=0, output_dim=0),
            has_target=False,
            variation=VariationConfig(enabled=True, num_entities=n_cols, reg_weight=0.0),
        ),
        "mc_entry": SyntheticRegressionType(
            name="mc_entry",
            slices=RegressionSlices(input_dim=in_dim, output_dim=1),
            has_target=True,
            variation=VariationConfig(enabled=False, num_entities=0, reg_weight=0.0),
        ),
    }


def _build_relationships() -> List[Relationship]:
    return [
        Relationship(name="entry_to_row", source_type="mc_entry", target_type="mc_row", inverse="row_to_entry"),
        Relationship(name="row_to_entry", source_type="mc_row", target_type="mc_entry", inverse="entry_to_row"),
        Relationship(name="entry_to_col", source_type="mc_entry", target_type="mc_col", inverse="col_to_entry"),
        Relationship(name="col_to_entry", source_type="mc_col", target_type="mc_entry", inverse="entry_to_col"),
    ]


def build_sample(
    *,
    n_rows: int,
    n_cols: int,
    rank: int,
    mask_prob: float,
    mask_mode: str,
    fixed_masked_i: int | None,
    fixed_masked_j: int | None,
    clip_min: float | None,
    clip_max: float | None,
    binned_targets: bool,
    num_bins: int,
    bin_min: float | None,
    bin_max: float | None,
    latent_sample_min: float,
    latent_sample_max: float,
    types: Dict[str, Any],
    relationships: List[Relationship],
    device: torch.device,
) -> Sample:
    if mask_mode not in ("random", "fixed", "random_one"):
        raise ValueError(f"mask_mode must be 'random', 'fixed', or 'random_one', got {mask_mode!r}")
    if mask_mode == "random":
        if not (0.0 < mask_prob < 1.0):
            raise ValueError("mask_prob must be in (0,1) for mask_mode=random")
    elif mask_mode == "fixed":
        if fixed_masked_i is None or fixed_masked_j is None:
            raise ValueError("mask_mode=fixed requires --fixed-mask-i and --fixed-mask-j")
        if not (0 <= int(fixed_masked_i) < n_rows and 0 <= int(fixed_masked_j) < n_cols):
            raise ValueError(f"fixed mask pair out of bounds: ({fixed_masked_i}, {fixed_masked_j}) for N={n_rows}, M={n_cols}")
    if latent_sample_max <= latent_sample_min:
        raise ValueError("latent_sample_max must be > latent_sample_min")
    if (clip_min is None) != (clip_max is None):
        raise ValueError("clip_min and clip_max must be both set or both None")
    if clip_min is not None and clip_max is not None and clip_max <= clip_min:
        raise ValueError("clip_max must be > clip_min")
    if (bin_min is None) != (bin_max is None):
        raise ValueError("bin_min and bin_max must be both set or both None")

    u = torch.empty(n_rows, rank, device=device).uniform_(latent_sample_min, latent_sample_max)
    v = torch.empty(n_cols, rank, device=device).uniform_(latent_sample_min, latent_sample_max)
    y = (u @ v.T).to(dtype=torch.float32)
    if clip_min is not None and clip_max is not None:
        y = y.clamp(min=float(clip_min), max=float(clip_max))
    if binned_targets:
        if bin_min is None or bin_max is None:
            dmin, dmax = _product_bounds(latent_sample_min, latent_sample_max)
        else:
            dmin, dmax = float(bin_min), float(bin_max)
        y = _quantize_to_bins(y, num_bins=num_bins, bin_min=dmin, bin_max=dmax)

    if mask_mode == "random":
        observed = (torch.rand(n_rows, n_cols, device=device) > mask_prob).bool()
    elif mask_mode == "fixed":
        observed = torch.ones(n_rows, n_cols, device=device, dtype=torch.bool)
        observed[int(fixed_masked_i), int(fixed_masked_j)] = False
    else:
        observed = torch.ones(n_rows, n_cols, device=device, dtype=torch.bool)
        miss_idx = int(torch.randint(0, n_rows * n_cols, (1,), device=device).item())
        miss_i, miss_j = divmod(miss_idx, n_cols)
        observed[miss_i, miss_j] = False

    tokens: List[Token] = []
    edges: List[Tuple[int, int, str]] = []

    row_start = 0
    for i in range(n_rows):
        tokens.append(Token(type_name="mc_row", entity_id=i, status=2, raw_data={}))
    col_start = len(tokens)
    for j in range(n_cols):
        tokens.append(Token(type_name="mc_col", entity_id=j, status=2, raw_data={}))

    entry_targets: List[float] = []
    entry_token_indices: List[int] = []
    masked_entry_positions: List[int] = []
    entry_pos = 0
    for i in range(n_rows):
        for j in range(n_cols):
            tgt = float(y[i, j].item())
            is_masked = not bool(observed[i, j].item())
            input_val = tgt if not is_masked else 0.0

            in_dim = types["mc_entry"].slices.input_dim
            raw: Dict[str, Any] = {"target_value": [tgt]}
            if in_dim == 2:
                raw["input_value"] = [input_val, 1.0 if is_masked else 0.0]
            else:
                raw["input_value"] = [input_val]

            entry_idx = len(tokens)
            status = 1 if is_masked else 2
            tokens.append(Token(type_name="mc_entry", entity_id=-1, status=status, raw_data=raw))
            entry_targets.append(tgt)
            entry_token_indices.append(entry_idx)
            if is_masked:
                masked_entry_positions.append(entry_pos)

            row_idx = row_start + i
            col_idx = col_start + j
            edges.append((entry_idx, row_idx, "entry_to_row"))
            edges.append((row_idx, entry_idx, "row_to_entry"))
            edges.append((entry_idx, col_idx, "entry_to_col"))
            edges.append((col_idx, entry_idx, "col_to_entry"))
            entry_pos += 1

    graph = EntityGraph(types=types, relationships=relationships, tokens=tokens, edges=edges)
    return Sample(
        graph=graph,
        entry_targets=torch.tensor(entry_targets, device=device, dtype=torch.float32),
        entry_token_indices=entry_token_indices,
        masked_entry_positions=masked_entry_positions,
    )


def _compute_mse_from_params(*, params: torch.Tensor, samples: List[Sample], masked_only: bool) -> torch.Tensor:
    # params: [B, L_total, P] -> mc_entry output dim is 1
    pred_rows: List[torch.Tensor] = []
    tgt_rows: List[torch.Tensor] = []
    for b, s in enumerate(samples):
        pred_rows.append(params[b, s.entry_token_indices, 0])
        tgt_rows.append(s.entry_targets)
    pred = torch.stack(pred_rows, dim=0)  # [B, L_entry]
    tgt = torch.stack(tgt_rows, dim=0)  # [B, L_entry]
    if not masked_only:
        return F.mse_loss(pred, tgt, reduction="mean")

    keep = torch.zeros_like(pred, dtype=torch.bool)
    for b, s in enumerate(samples):
        keep[b, s.masked_entry_positions] = True
    return F.mse_loss(pred[keep], tgt[keep], reduction="mean") if keep.any() else torch.zeros((), device=pred.device)


def render_plots(results: Dict[str, Any], out_dir: Path, *, log_y: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = np.asarray(results["train_mse"], dtype=np.float64)
    te = np.asarray(results["test_mse"], dtype=np.float64)
    x = np.arange(1, tr.shape[0] + 1)
    if log_y:
        tr = np.maximum(tr, _LOG_Y_FLOOR)
        te = np.maximum(te, _LOG_Y_FLOOR)
    for y, title, fname in (
        (tr, "Train MSE (all entries)", "rowcol_tokens_train_mse.png"),
        (te, "Test MSE (masked entries)", "rowcol_tokens_test_masked_mse.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        ax.plot(x, y, linewidth=2.0)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("MSE")
        ax.set_title(title + (" (log y)" if log_y else ""))
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def run(
    *,
    out_dir: Path,
    seed: int,
    steps: int,
    n_rows: int,
    n_cols: int,
    rank: int,
    mask_prob: float,
    mask_mode: str,
    fixed_masked_i: int | None,
    fixed_masked_j: int | None,
    clip_min: float | None,
    clip_max: float | None,
    binned_targets: bool,
    num_bins: int,
    bin_min: float | None,
    bin_max: float | None,
    latent_sample_min: float,
    latent_sample_max: float,
    train_batch_size: int,
    eval_batch_size: int,
    embedding_dim: int,
    num_layers: int,
    attn_heads: int,
    d_ff: int,
    num_ffn_layers: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    type_embedding_init: str,
    eval_every: int,
    live_curves_every: int,
    log_every: int,
    use_mc_mask_bit: bool,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(seed)

    types = _build_types(n_rows=n_rows, n_cols=n_cols, use_mask_bit=use_mc_mask_bit)
    relationships = _build_relationships()

    ref = build_sample(
        n_rows=n_rows,
        n_cols=n_cols,
        rank=rank,
        mask_prob=mask_prob,
        mask_mode=mask_mode,
        fixed_masked_i=fixed_masked_i,
        fixed_masked_j=fixed_masked_j,
        clip_min=clip_min,
        clip_max=clip_max,
        binned_targets=binned_targets,
        num_bins=num_bins,
        bin_min=bin_min,
        bin_max=bin_max,
        latent_sample_min=latent_sample_min,
        latent_sample_max=latent_sample_max,
        types=types,
        relationships=relationships,
        device=device,
    )

    cfg = EntityMarformerConfig(
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        attention_heads=attn_heads,
        dropout=dropout,
        d_ff=d_ff,
        num_ffn_layers=num_ffn_layers,
        use_per_head_rel=False,
        use_pointer=False,
        use_rel_value=False,
        use_addone_attn=False,
        type_embedding_init=type_embedding_init,
        use_deviation_norm=False,
        scale_shared_rel=False,
        use_learned_embedding=False,
        use_graph_mask=False,
        use_multiplication_head=False,
    )
    model = EntityMarformer(config=cfg, types=types, num_relationships=ref.graph.num_relationships).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_curve: List[float] = []
    test_curve: List[float] = []

    def _payload(*, completed_steps: int, live: bool) -> Dict[str, Any]:
        return {
            "seed": seed,
            "num_steps": steps,
            "completed_steps": completed_steps,
            "live": live,
            "N": n_rows,
            "M": n_cols,
            "rank": rank,
            "mask_prob": mask_prob,
            "mask_mode": mask_mode,
            "fixed_masked_i": fixed_masked_i,
            "fixed_masked_j": fixed_masked_j,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "binned_targets": binned_targets,
            "num_bins": num_bins,
            "bin_min": bin_min,
            "bin_max": bin_max,
            "latent_sample_min": latent_sample_min,
            "latent_sample_max": latent_sample_max,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "attn_heads": attn_heads,
            "d_ff": d_ff,
            "num_ffn_layers": num_ffn_layers,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "type_embedding_init": type_embedding_init,
            "eval_every": eval_every,
            "live_curves_every": live_curves_every,
            "log_every": log_every,
            "use_mc_mask_bit": use_mc_mask_bit,
            "mc_row_variation": True,
            "mc_col_variation": True,
            "num_relationships": ref.graph.num_relationships,
            "train_mse": train_curve,
            "test_mse": test_curve,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    if live_curves_every > 0:
        (out_dir / "curves_live.json").unlink(missing_ok=True)

    last_test: float | None = None
    it = tqdm(range(steps), total=steps, desc="train", leave=False)
    for step in it:
        train_samples = [
            build_sample(
                n_rows=n_rows,
                n_cols=n_cols,
                rank=rank,
                mask_prob=mask_prob,
                mask_mode=mask_mode,
                fixed_masked_i=fixed_masked_i,
                fixed_masked_j=fixed_masked_j,
                clip_min=clip_min,
                clip_max=clip_max,
                binned_targets=binned_targets,
                num_bins=num_bins,
                bin_min=bin_min,
                bin_max=bin_max,
                latent_sample_min=latent_sample_min,
                latent_sample_max=latent_sample_max,
                types=types,
                relationships=relationships,
                device=device,
            )
            for _ in range(train_batch_size)
        ]

        model.train()
        opt.zero_grad(set_to_none=True)
        p_tr = model.forward_batch([s.graph for s in train_samples], device=device)
        train_mse = _compute_mse_from_params(params=p_tr, samples=train_samples, masked_only=False)

        done = step + 1
        will_log = log_every > 0 and (done == 1 or done % log_every == 0 or done == steps)
        if will_log:
            with torch.no_grad():
                train_masked_log = float(
                    _compute_mse_from_params(params=p_tr, samples=train_samples, masked_only=True).detach().cpu().item()
                )
                train_all_log = float(train_mse.detach().cpu().item())

        train_mse.backward()
        opt.step()

        base_eval = (step == 0) or ((step + 1) % max(1, eval_every) == 0) or (step == steps - 1)
        do_eval = base_eval or will_log
        if do_eval:
            test_samples = [
                build_sample(
                    n_rows=n_rows,
                    n_cols=n_cols,
                    rank=rank,
                    mask_prob=mask_prob,
                    mask_mode=mask_mode,
                    fixed_masked_i=fixed_masked_i,
                    fixed_masked_j=fixed_masked_j,
                    clip_min=clip_min,
                    clip_max=clip_max,
                    binned_targets=binned_targets,
                    num_bins=num_bins,
                    bin_min=bin_min,
                    bin_max=bin_max,
                    latent_sample_min=latent_sample_min,
                    latent_sample_max=latent_sample_max,
                    types=types,
                    relationships=relationships,
                    device=device,
                )
                for _ in range(eval_batch_size)
            ]
            model.eval()
            with torch.no_grad():
                p_te = model.forward_batch([s.graph for s in test_samples], device=device)
                test_mse = _compute_mse_from_params(params=p_te, samples=test_samples, masked_only=True)
            last_test = float(test_mse.detach().cpu().item())
        else:
            assert last_test is not None
            test_mse = torch.tensor(last_test, device=device)

        train_curve.append(float(train_mse.detach().cpu().item()))
        test_curve.append(float(test_mse.detach().cpu().item()))

        if will_log:
            print(
                f"step {done:5d} | train_all={train_all_log:.6f} "
                f"train_masked={train_masked_log:.6f} eval_masked={float(test_mse.detach().cpu().item()):.6f}"
            )

        if live_curves_every > 0 and (done == 1 or done % live_curves_every == 0 or done == steps):
            _atomic_write_json(out_dir / "curves_live.json", _payload(completed_steps=done, live=True))

    results = _payload(completed_steps=steps, live=False)
    (out_dir / "curves.json").write_text(json.dumps(results, indent=2))
    if live_curves_every > 0:
        _atomic_write_json(out_dir / "curves_live.json", results)
    render_plots(results, out_dir, log_y=True)
    print(f"Wrote curves and plots to {out_dir}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="EntityMF with mc_row/mc_col entity tokens + variation, no fixed one-hot positional features.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--steps", type=int, default=NUM_STEPS)
    p.add_argument("--N", type=int, default=N_ROWS)
    p.add_argument("--M", type=int, default=N_COLS)
    p.add_argument("--rank", type=int, default=RANK)
    p.add_argument("--mask-prob", type=float, default=MASK_PROB)
    p.add_argument(
        "--mask-mode",
        type=str,
        default="random",
        choices=("random", "fixed", "random_one"),
        help=(
            "random: MCAR masking via mask_prob; "
            "fixed: always mask exactly one cell at (--fixed-mask-i,--fixed-mask-j); "
            "random_one: exactly one uniformly random masked cell each sample."
        ),
    )
    p.add_argument("--fixed-mask-i", type=int, default=None)
    p.add_argument("--fixed-mask-j", type=int, default=None)
    p.add_argument("--clip-min", type=float, default=None)
    p.add_argument("--clip-max", type=float, default=None)
    p.add_argument("--binned-targets", action="store_true", help="Quantize targets into bins before masking/loss.")
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--bin-min", type=float, default=None)
    p.add_argument("--bin-max", type=float, default=None)
    p.add_argument("--latent-sample-min", type=float, default=LATENT_SAMPLE_MIN)
    p.add_argument("--latent-sample-max", type=float, default=LATENT_SAMPLE_MAX)
    p.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    p.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--attn-heads", type=int, default=ATTN_HEADS)
    p.add_argument("--d-ff", type=int, default=D_FF)
    p.add_argument("--num-ffn-layers", type=int, default=NUM_FFN_LAYERS)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--type-embedding-init", type=str, default=TYPE_EMBEDDING_INIT, choices=("normal", "scaled_normal", "kaiming"))
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--live-curves-every", type=int, default=LIVE_CURVES_EVERY)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--use-mc-mask-bit", action="store_true", help="Use input_value=[value, is_masked] instead of [value].")
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    run(
        out_dir=out,
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
        binned_targets=args.binned_targets,
        num_bins=args.num_bins,
        bin_min=args.bin_min,
        bin_max=args.bin_max,
        latent_sample_min=args.latent_sample_min,
        latent_sample_max=args.latent_sample_max,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        attn_heads=args.attn_heads,
        d_ff=args.d_ff,
        num_ffn_layers=args.num_ffn_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        type_embedding_init=args.type_embedding_init,
        eval_every=args.eval_every,
        live_curves_every=args.live_curves_every,
        log_every=args.log_every,
        use_mc_mask_bit=args.use_mc_mask_bit,
    )


if __name__ == "__main__":
    main()

