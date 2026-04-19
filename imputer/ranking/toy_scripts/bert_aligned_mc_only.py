#!/usr/bin/env python3
from __future__ import annotations

"""
BERT-aligned matrix completion (mc_entry-only) with an explicit mask bit.

Goal: isolate whether EntityMarformer’s non-grokking is due to graph/entity design,
by running the *same* 4x4 rank-1 toy task through a standard Transformer encoder,
with an input representation that matches the BERT toy as closely as possible:

  h_tok[..., 0] = observed value (masked -> 0)
  h_tok[..., 1] = is_masked bit (masked -> 1, observed -> 0)
  h_tok[..., 2:] = 0
  h = h_tok + pos_emb

No relational bias, no entity tokens, no edge masks: a vanilla encoder-only Transformer.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


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

HIDDEN_SIZE = 768
NUM_LAYERS = 4
NUM_HEADS = 8
INTERMEDIATE_SIZE = 3072
DROPOUT = 0.0

LR = 1e-4
WEIGHT_DECAY = 0.0

POSITIONAL_SCHEME = "flat"  # flat | rowcol_concat

LOG_EVERY = 50
EVAL_EVERY = 50
LIVE_CURVES_EVERY = 0

OUT_DIR = Path("OUTPUT/bert_aligned_mc_only")
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
class Batch:
    input_values: torch.Tensor  # [B, L] observed value, masked -> 0
    mask_bits: torch.Tensor  # [B, L] masked -> 1, observed -> 0
    target: torch.Tensor  # [B, N, M]
    observed_mask: torch.Tensor  # [B, N, M], bool


def sample_low_rank_batch(
    *,
    batch_size: int,
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
    device: torch.device,
) -> Batch:
    if mask_mode not in ("random", "fixed", "random_one"):
        raise ValueError(f"mask_mode must be 'random', 'fixed', or 'random_one', got {mask_mode!r}")
    if mask_mode == "random":
        if not (0.0 < mask_prob < 1.0):
            raise ValueError(f"mask_prob must be in (0,1) for mask_mode=random, got {mask_prob}")
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

    u = torch.empty(batch_size, n_rows, rank, device=device).uniform_(latent_sample_min, latent_sample_max)
    v = torch.empty(batch_size, n_cols, rank, device=device).uniform_(latent_sample_min, latent_sample_max)
    x = torch.matmul(u, v.transpose(-1, -2))  # [B, N, M]
    target = x.to(dtype=torch.float32)
    if clip_min is not None and clip_max is not None:
        target = target.clamp(min=float(clip_min), max=float(clip_max))
    if binned_targets:
        if bin_min is None or bin_max is None:
            dmin, dmax = _product_bounds(latent_sample_min, latent_sample_max)
        else:
            dmin, dmax = float(bin_min), float(bin_max)
        target = _quantize_to_bins(target, num_bins=num_bins, bin_min=dmin, bin_max=dmax)

    if mask_mode == "random":
        observed = (torch.rand(batch_size, n_rows, n_cols, device=device) > mask_prob).bool()
    elif mask_mode == "fixed":
        observed = torch.ones(batch_size, n_rows, n_cols, device=device, dtype=torch.bool)
        observed[:, int(fixed_masked_i), int(fixed_masked_j)] = False
    else:
        observed = torch.ones(batch_size, n_rows, n_cols, device=device, dtype=torch.bool)
        miss_idx = torch.randint(0, n_rows * n_cols, (batch_size,), device=device)
        miss_i = torch.div(miss_idx, n_cols, rounding_mode="floor")
        miss_j = miss_idx % n_cols
        observed[torch.arange(batch_size, device=device), miss_i, miss_j] = False
    input_values = torch.where(observed, target, torch.zeros_like(target))
    mask_bits = (~observed).to(dtype=torch.float32)

    return Batch(
        input_values=input_values.reshape(batch_size, n_rows * n_cols),
        mask_bits=mask_bits.reshape(batch_size, n_rows * n_cols),
        target=target,
        observed_mask=observed,
    )


class BertAlignedMcOnlyModel(nn.Module):
    def __init__(
        self,
        *,
        n_rows: int,
        n_cols: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        positional_scheme: str = POSITIONAL_SCHEME,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 2:
            raise ValueError("hidden_size must be >= 2 to hold [value, mask_bit]")
        if positional_scheme not in ("flat", "rowcol_concat"):
            raise ValueError(f"Unknown positional_scheme={positional_scheme!r}")

        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.seq_len = self.n_rows * self.n_cols
        self.hidden_size = int(hidden_size)
        self.positional_scheme = positional_scheme

        if self.positional_scheme == "flat":
            self.pos_emb = nn.Embedding(self.seq_len, self.hidden_size)
        else:
            row_dim = self.hidden_size // 2
            col_dim = self.hidden_size - row_dim
            self.row_pos_emb = nn.Embedding(self.n_rows, row_dim)
            self.col_pos_emb = nn.Embedding(self.n_cols, col_dim)

        self.input_norm = nn.LayerNorm(self.hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_head = nn.Linear(self.hidden_size, 1)

    def _build_positional_embeddings(self, bsz: int, device: torch.device) -> torch.Tensor:
        if self.positional_scheme == "flat":
            pos = torch.arange(self.seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            return self.pos_emb(pos)
        row_ids = torch.arange(self.n_rows, device=device).repeat_interleave(self.n_cols)
        col_ids = torch.arange(self.n_cols, device=device).repeat(self.n_rows)
        row_ids = row_ids.unsqueeze(0).expand(bsz, -1)
        col_ids = col_ids.unsqueeze(0).expand(bsz, -1)
        return torch.cat([self.row_pos_emb(row_ids), self.col_pos_emb(col_ids)], dim=-1)

    def forward(self, *, input_values: torch.Tensor, mask_bits: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_values.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        if mask_bits.shape != (bsz, seq_len):
            raise ValueError("mask_bits must match input_values shape")

        h_tok = torch.zeros(bsz, seq_len, self.hidden_size, device=input_values.device, dtype=torch.float32)
        h_tok[..., 0] = input_values.to(dtype=h_tok.dtype)
        h_tok[..., 1] = mask_bits.to(dtype=h_tok.dtype)
        h = h_tok + self._build_positional_embeddings(bsz, input_values.device)
        h = self.input_norm(h)
        h = self.encoder(h)
        return self.out_head(h).squeeze(-1)  # [B, L]


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sel = mask.bool()
    if sel.any():
        return F.mse_loss(pred[sel], target[sel], reduction="mean")
    return torch.zeros((), dtype=pred.dtype, device=pred.device)


def compute_losses(
    *,
    pred: torch.Tensor,  # [B, N, M]
    target: torch.Tensor,  # [B, N, M]
    observed_mask: torch.Tensor,  # [B, N, M]
) -> Dict[str, torch.Tensor]:
    total = F.mse_loss(pred, target, reduction="mean")
    obs = _masked_mse(pred, target, observed_mask)
    masked = _masked_mse(pred, target, ~observed_mask)
    return {"total_mse": total, "observed_mse": obs, "masked_mse": masked}


def render_plots_from_results(results: Dict[str, Any], out_dir: Path, *, log_y: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = np.asarray(results["train_total_mse"], dtype=np.float64)
    te = np.asarray(results["eval_masked_mse"], dtype=np.float64)
    x = np.arange(1, tr.shape[0] + 1)
    if log_y:
        tr = np.maximum(tr, _LOG_Y_FLOOR)
        te = np.maximum(te, _LOG_Y_FLOOR)

    for y, title, fname in (
        (tr, "Train Total MSE", "mc_only_train_total_mse.png"),
        (te, "Eval Masked-entry MSE", "mc_only_eval_masked_mse.png"),
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


def run_experiment(
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
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    intermediate_size: int,
    positional_scheme: str,
    dropout: float,
    lr: float,
    weight_decay: float,
    log_every: int,
    eval_every: int,
    live_curves_every: int,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(seed)

    model = BertAlignedMcOnlyModel(
        n_rows=n_rows,
        n_cols=n_cols,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        positional_scheme=positional_scheme,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    curves: Dict[str, List[float]] = {
        "train_total_mse": [],
        "eval_total_mse": [],
        "eval_masked_mse": [],
        "eval_observed_mse": [],
    }

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
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "positional_scheme": positional_scheme,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            **curves,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    if live_curves_every > 0:
        (out_dir / "curves_live.json").unlink(missing_ok=True)

    last_eval: Dict[str, torch.Tensor] | None = None
    it = tqdm(range(steps), total=steps, desc="train", leave=False)
    for step in it:
        model.train()
        opt.zero_grad(set_to_none=True)
        b_tr = sample_low_rank_batch(
            batch_size=train_batch_size,
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
            device=device,
        )
        pred = model(input_values=b_tr.input_values, mask_bits=b_tr.mask_bits).reshape(
            train_batch_size, n_rows, n_cols
        )
        train_metrics = compute_losses(pred=pred, target=b_tr.target, observed_mask=b_tr.observed_mask)
        train_metrics["total_mse"].backward()
        opt.step()

        do_eval = (step == 0) or ((step + 1) % max(1, eval_every) == 0) or (step == steps - 1)
        if do_eval:
            model.eval()
            with torch.no_grad():
                b_ev = sample_low_rank_batch(
                    batch_size=eval_batch_size,
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
                    device=device,
                )
                pred_ev = model(input_values=b_ev.input_values, mask_bits=b_ev.mask_bits).reshape(
                    eval_batch_size, n_rows, n_cols
                )
                eval_metrics = compute_losses(pred=pred_ev, target=b_ev.target, observed_mask=b_ev.observed_mask)
            last_eval = eval_metrics
        else:
            assert last_eval is not None
            eval_metrics = last_eval

        curves["train_total_mse"].append(float(train_metrics["total_mse"].detach().cpu().item()))
        curves["eval_total_mse"].append(float(eval_metrics["total_mse"].detach().cpu().item()))
        curves["eval_masked_mse"].append(float(eval_metrics["masked_mse"].detach().cpu().item()))
        curves["eval_observed_mse"].append(float(eval_metrics["observed_mse"].detach().cpu().item()))

        done = step + 1
        if live_curves_every > 0 and (done == 1 or done % live_curves_every == 0 or done == steps):
            _atomic_write_json(out_dir / "curves_live.json", _payload(completed_steps=done, live=True))

        if done % max(1, log_every) == 0 or step == 0:
            print(
                f"step {done:5d} | train_total={curves['train_total_mse'][-1]:.6f} "
                f"eval_masked={curves['eval_masked_mse'][-1]:.6f} eval_total={curves['eval_total_mse'][-1]:.6f}"
            )

    results = _payload(completed_steps=steps, live=False)
    (out_dir / "curves.json").write_text(json.dumps(results, indent=2))
    if live_curves_every > 0:
        _atomic_write_json(out_dir / "curves_live.json", results)
    render_plots_from_results(results, out_dir, log_y=True)
    print(f"Wrote curves and plots to {out_dir}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="BERT-aligned mc_entry-only matrix completion (with mask bit).")
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
    p.add_argument("--latent-sample-min", type=float, default=LATENT_SAMPLE_MIN)
    p.add_argument("--latent-sample-max", type=float, default=LATENT_SAMPLE_MAX)
    p.add_argument("--clip-min", type=float, default=None)
    p.add_argument("--clip-max", type=float, default=None)
    p.add_argument("--binned-targets", action="store_true", help="Quantize targets into bins before masking/loss.")
    p.add_argument("--num-bins", type=int, default=5)
    p.add_argument("--bin-min", type=float, default=None)
    p.add_argument("--bin-max", type=float, default=None)
    p.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--num-heads", type=int, default=NUM_HEADS)
    p.add_argument("--intermediate-size", type=int, default=INTERMEDIATE_SIZE)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--live-curves-every", type=int, default=LIVE_CURVES_EVERY)
    p.add_argument(
        "--positional-scheme",
        type=str,
        default=POSITIONAL_SCHEME,
        choices=("flat", "rowcol_concat"),
    )
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    run_experiment(
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
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        positional_scheme=args.positional_scheme,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        eval_every=args.eval_every,
        live_curves_every=args.live_curves_every,
    )


if __name__ == "__main__":
    main()

