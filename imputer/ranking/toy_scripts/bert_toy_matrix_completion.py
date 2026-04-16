#!/usr/bin/env python3
from __future__ import annotations

"""
BERT-style toy matrix completion inspired by:
"Abrupt Learning in Transformers: A Case Study on Matrix Completion" (arXiv:2410.22244).

This script follows the paper/repo semantics:
- encoder-only Transformer with learned absolute positions
- no token-type embeddings
- no dropout
- scalar readout per token position, trained with MSE
- online synthetic low-rank sampling each step

Minor implementation deviation:
- Uses a minimal PyTorch Transformer encoder instead of HuggingFace BERT internals.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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

N_ROWS = 7
N_COLS = 7
RANK = 2
MASK_PROB = 0.3

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 64

MIN_VOCAB = -10.0
MAX_VOCAB = 10.0
PREC = 2

HIDDEN_SIZE = 768
NUM_LAYERS = 4
NUM_HEADS = 8
INTERMEDIATE_SIZE = 3072
DROPOUT = 0.0

LR = 1e-4
WEIGHT_DECAY = 0.0
POSITIONAL_SCHEME = "flat"  # choices: flat, rowcol_concat
CONTINUOUS_TOKENIZATION = False

LOG_EVERY = 50
EVAL_EVERY = 50
DEBUG_EVERY = 500
LIVE_CURVES_EVERY = 0

OUT_DIR = Path("OUTPUT/bert_toy_matrix_completion")
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


class NumericTokenizer:
    """Maps rounded numeric values to token ids. token_id=0 is MASK."""

    def __init__(self, *, min_vocab: float, max_vocab: float, prec: int) -> None:
        self.min_vocab = float(min_vocab)
        self.max_vocab = float(max_vocab)
        self.prec = int(prec)
        self.scale = 10**self.prec
        self.min_int = int(round(self.min_vocab * self.scale))
        self.max_int = int(round(self.max_vocab * self.scale))
        if self.max_int < self.min_int:
            raise ValueError("max_vocab must be >= min_vocab")
        self.num_value_tokens = self.max_int - self.min_int + 1
        self.vocab_size = 1 + self.num_value_tokens
        self.mask_token_id = 0

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * self.scale) / self.scale

    def values_to_token_ids(self, x: torch.Tensor) -> torch.Tensor:
        q = self.quantize(x)
        q_int = torch.round(q * self.scale).to(dtype=torch.long)
        q_int = torch.clamp(q_int, min=self.min_int, max=self.max_int)
        return (q_int - self.min_int + 1).to(dtype=torch.long)

    def token_ids_to_values(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(dtype=torch.long)
        val_ids = torch.clamp(ids, min=1, max=self.vocab_size - 1) - 1
        q_int = val_ids + self.min_int
        return q_int.to(dtype=torch.float32) / float(self.scale)


@dataclass
class Batch:
    token_ids: torch.Tensor  # [B, L]
    input_values: torch.Tensor  # [B, L], observed value or 0 for masked
    target: torch.Tensor  # [B, N, M]
    observed_mask: torch.Tensor  # [B, N, M], bool


def sample_low_rank_batch(
    *,
    batch_size: int,
    n_rows: int,
    n_cols: int,
    rank: int,
    mask_prob: float,
    tokenizer: NumericTokenizer,
    device: torch.device,
) -> Batch:
    if not (0.0 < mask_prob < 1.0):
        raise ValueError(f"mask_prob must be in (0,1), got {mask_prob}")
    u = torch.empty(batch_size, n_rows, rank, device=device).uniform_(-1.0, 1.0)
    v = torch.empty(batch_size, n_cols, rank, device=device).uniform_(-1.0, 1.0)
    x = torch.matmul(u, v.transpose(-1, -2))  # [B, N, M]
    target = tokenizer.quantize(x).to(dtype=torch.float32)

    observed = (torch.rand(batch_size, n_rows, n_cols, device=device) > mask_prob).bool()
    value_ids = tokenizer.values_to_token_ids(target)  # [B, N, M]
    input_values = torch.where(
        observed,
        target,
        torch.zeros_like(target, dtype=torch.float32, device=device),
    )
    token_ids = torch.where(
        observed,
        value_ids,
        torch.zeros_like(value_ids, dtype=torch.long, device=device),
    )
    token_ids = token_ids.reshape(batch_size, n_rows * n_cols)
    input_values = input_values.reshape(batch_size, n_rows * n_cols)
    return Batch(token_ids=token_ids, input_values=input_values, target=target, observed_mask=observed)


class BertMatrixCompletionModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        n_rows: int,
        n_cols: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        positional_scheme: str = POSITIONAL_SCHEME,
        continuous_tokenization: bool = CONTINUOUS_TOKENIZATION,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if positional_scheme not in ("flat", "rowcol_concat"):
            raise ValueError(f"Unknown positional_scheme={positional_scheme!r}")
        self.seq_len = seq_len
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.positional_scheme = positional_scheme
        self.continuous_tokenization = continuous_tokenization
        if self.n_rows * self.n_cols != self.seq_len:
            raise ValueError(f"Expected n_rows*n_cols == seq_len, got {self.n_rows}*{self.n_cols} vs {self.seq_len}")

        if not self.continuous_tokenization:
            self.token_emb = nn.Embedding(vocab_size, hidden_size)

        if self.positional_scheme == "flat":
            self.pos_emb = nn.Embedding(seq_len, hidden_size)
        else:
            # Positional embedding is [row_abs ; col_abs] concatenated.
            row_dim = hidden_size // 2
            col_dim = hidden_size - row_dim
            self.row_pos_emb = nn.Embedding(n_rows, row_dim)
            self.col_pos_emb = nn.Embedding(n_cols, col_dim)
        self.input_norm = nn.LayerNorm(hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_head = nn.Linear(hidden_size, 1)

    def _build_positional_embeddings(self, bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.positional_scheme == "flat":
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            return self.pos_emb(pos)

        row_ids = torch.arange(self.n_rows, device=device).repeat_interleave(self.n_cols)
        col_ids = torch.arange(self.n_cols, device=device).repeat(self.n_rows)
        row_ids = row_ids.unsqueeze(0).expand(bsz, -1)
        col_ids = col_ids.unsqueeze(0).expand(bsz, -1)
        row_emb = self.row_pos_emb(row_ids)
        col_emb = self.col_pos_emb(col_ids)
        return torch.cat([row_emb, col_emb], dim=-1)

    def _build_token_embeddings(
        self,
        *,
        token_ids: torch.Tensor | None,
        input_values: torch.Tensor | None,
        bsz: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.continuous_tokenization:
            if input_values is None:
                raise ValueError("input_values is required when continuous_tokenization=True")
            if input_values.shape != (bsz, seq_len):
                raise ValueError(f"Expected input_values shape {(bsz, seq_len)}, got {tuple(input_values.shape)}")
            h_tok = torch.zeros(bsz, seq_len, self.out_head.in_features, device=device, dtype=torch.float32)
            h_tok[..., 0] = input_values.to(dtype=h_tok.dtype)
            return h_tok

        if token_ids is None:
            raise ValueError("token_ids is required when continuous_tokenization=False")
        return self.token_emb(token_ids)

    def forward(
        self,
        *,
        token_ids: torch.Tensor | None = None,
        input_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape_source = input_values if (self.continuous_tokenization and input_values is not None) else token_ids
        if shape_source is None:
            raise ValueError("At least one of token_ids/input_values must be provided")
        bsz, seq_len = shape_source.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        h_tok = self._build_token_embeddings(
            token_ids=token_ids,
            input_values=input_values,
            bsz=bsz,
            seq_len=seq_len,
            device=shape_source.device,
        )
        h_pos = self._build_positional_embeddings(bsz, seq_len, shape_source.device)
        h = h_tok + h_pos
        h = self.input_norm(h)
        h = self.encoder(h)
        y = self.out_head(h).squeeze(-1)  # [B, L]
        return y


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
    masked_abs = torch.abs(pred[~observed_mask]).mean() if (~observed_mask).any() else torch.zeros_like(total)
    return {
        "total_mse": total,
        "observed_mse": obs,
        "masked_mse": masked,
        "masked_abs_pred_mean": masked_abs,
    }


def _pretty_print_example(*, pred: torch.Tensor, target: torch.Tensor, observed_mask: torch.Tensor) -> None:
    n_rows, n_cols = target.shape
    print("  debug: example target vs prediction (* = masked)")
    hdr = "\t".join(f"j={j}".rjust(8) for j in range(n_cols))
    print(f"  \t{hdr}\t||\t{hdr}")
    for i in range(n_rows):
        left = []
        right = []
        for j in range(n_cols):
            mark = " " if observed_mask[i, j] else "*"
            left.append(f"{target[i, j]:>7.2f}{mark}")
            right.append(f"{pred[i, j]:>8.2f}")
        print(f"  i={i}\t" + "\t".join(left) + "\t||\t" + "\t".join(right))


def render_plots_from_results(results: Dict[str, Any], out_dir: Path, *, log_y: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    series = {
        "train_total_mse": np.asarray(results["train_total_mse"], dtype=np.float64),
        "eval_total_mse": np.asarray(results["eval_total_mse"], dtype=np.float64),
        "train_masked_mse": np.asarray(results["train_masked_mse"], dtype=np.float64),
        "train_observed_mse": np.asarray(results["train_observed_mse"], dtype=np.float64),
        "eval_masked_mse": np.asarray(results["eval_masked_mse"], dtype=np.float64),
        "eval_observed_mse": np.asarray(results["eval_observed_mse"], dtype=np.float64),
    }
    n_plot = len(series["train_total_mse"])
    x = np.arange(1, n_plot + 1)
    for k in list(series):
        if log_y:
            series[k] = np.maximum(series[k], _LOG_Y_FLOOR)

    for key, title, fname in (
        ("train_total_mse", "Train Total MSE", "bert_mc_train_total_mse.png"),
        ("eval_total_mse", "Eval Total MSE", "bert_mc_eval_total_mse.png"),
        ("train_masked_mse", "Train Masked-entry MSE", "bert_mc_train_masked_mse.png"),
        ("eval_masked_mse", "Eval Masked-entry MSE", "bert_mc_eval_masked_mse.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        ax.plot(x, series[key], linewidth=2.0)
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
    out_dir: Path = OUT_DIR,
    seed: int = SEED,
    steps: int = NUM_STEPS,
    n_rows: int = N_ROWS,
    n_cols: int = N_COLS,
    rank: int = RANK,
    mask_prob: float = MASK_PROB,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    min_vocab: float = MIN_VOCAB,
    max_vocab: float = MAX_VOCAB,
    prec: int = PREC,
    hidden_size: int = HIDDEN_SIZE,
    num_layers: int = NUM_LAYERS,
    num_heads: int = NUM_HEADS,
    intermediate_size: int = INTERMEDIATE_SIZE,
    positional_scheme: str = POSITIONAL_SCHEME,
    continuous_tokenization: bool = CONTINUOUS_TOKENIZATION,
    dropout: float = DROPOUT,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    log_every: int = LOG_EVERY,
    eval_every: int = EVAL_EVERY,
    debug_every: int = DEBUG_EVERY,
    live_curves_every: int = LIVE_CURVES_EVERY,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(seed)

    tokenizer = NumericTokenizer(min_vocab=min_vocab, max_vocab=max_vocab, prec=prec)
    seq_len = n_rows * n_cols
    model = BertMatrixCompletionModel(
        vocab_size=tokenizer.vocab_size,
        n_rows=n_rows,
        n_cols=n_cols,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        positional_scheme=positional_scheme,
        continuous_tokenization=continuous_tokenization,
        dropout=dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(
        "Starting BERT toy matrix-completion experiment\n"
        f"  shape: N={n_rows}, M={n_cols}, rank={rank}, mask_prob={mask_prob}\n"
        f"  train_batch_size={train_batch_size}, eval_batch_size={eval_batch_size}, steps={steps}\n"
        f"  tokenizer: min_vocab={min_vocab}, max_vocab={max_vocab}, prec={prec}, vocab={tokenizer.vocab_size}\n"
        f"  positional_scheme={positional_scheme}, continuous_tokenization={continuous_tokenization}\n"
        f"  model: hidden={hidden_size}, layers={num_layers}, heads={num_heads}, ff={intermediate_size}, dropout={dropout}\n"
        f"  optim: Adam lr={lr}, weight_decay={weight_decay}\n"
        f"  device={device}"
    )

    curves: Dict[str, List[float]] = {
        "train_total_mse": [],
        "train_masked_mse": [],
        "train_observed_mse": [],
        "train_masked_abs_pred_mean": [],
        "eval_total_mse": [],
        "eval_masked_mse": [],
        "eval_observed_mse": [],
        "eval_masked_abs_pred_mean": [],
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
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "min_vocab": min_vocab,
            "max_vocab": max_vocab,
            "prec": prec,
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "positional_scheme": positional_scheme,
            "continuous_tokenization": continuous_tokenization,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            **curves,
            "train_mse": list(curves["train_total_mse"]),
            "test_mse": list(curves["eval_masked_mse"]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    if live_curves_every > 0:
        (out_dir / "curves_live.json").unlink(missing_ok=True)

    last_eval_metrics: Dict[str, torch.Tensor] | None = None
    steps_it = tqdm(range(steps), total=steps, desc="train", leave=False)
    for step in steps_it:
        model.train()
        opt.zero_grad(set_to_none=True)
        b_tr = sample_low_rank_batch(
            batch_size=train_batch_size,
            n_rows=n_rows,
            n_cols=n_cols,
            rank=rank,
            mask_prob=mask_prob,
            tokenizer=tokenizer,
            device=device,
        )
        pred_flat = model(token_ids=b_tr.token_ids, input_values=b_tr.input_values)
        pred = pred_flat.reshape(train_batch_size, n_rows, n_cols)
        train_metrics = compute_losses(pred=pred, target=b_tr.target, observed_mask=b_tr.observed_mask)
        train_metrics["total_mse"].backward()
        opt.step()

        b_ev: Batch | None = None
        pred_ev: torch.Tensor | None = None
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
                    tokenizer=tokenizer,
                    device=device,
                )
                pred_ev = model(token_ids=b_ev.token_ids, input_values=b_ev.input_values).reshape(
                    eval_batch_size, n_rows, n_cols
                )
                eval_metrics = compute_losses(pred=pred_ev, target=b_ev.target, observed_mask=b_ev.observed_mask)
            last_eval_metrics = eval_metrics
        else:
            assert last_eval_metrics is not None
            eval_metrics = last_eval_metrics

        curves["train_total_mse"].append(float(train_metrics["total_mse"].detach().cpu().item()))
        curves["train_masked_mse"].append(float(train_metrics["masked_mse"].detach().cpu().item()))
        curves["train_observed_mse"].append(float(train_metrics["observed_mse"].detach().cpu().item()))
        curves["train_masked_abs_pred_mean"].append(float(train_metrics["masked_abs_pred_mean"].detach().cpu().item()))
        curves["eval_total_mse"].append(float(eval_metrics["total_mse"].detach().cpu().item()))
        curves["eval_masked_mse"].append(float(eval_metrics["masked_mse"].detach().cpu().item()))
        curves["eval_observed_mse"].append(float(eval_metrics["observed_mse"].detach().cpu().item()))
        curves["eval_masked_abs_pred_mean"].append(float(eval_metrics["masked_abs_pred_mean"].detach().cpu().item()))

        done = step + 1
        if live_curves_every > 0 and (done == 1 or done % live_curves_every == 0 or done == steps):
            _atomic_write_json(out_dir / "curves_live.json", _payload(completed_steps=done, live=True))

        if done % max(1, log_every) == 0 or step == 0:
            print(
                f"step {done:5d} | "
                f"train_total={curves['train_total_mse'][-1]:.6f} "
                f"eval_total={curves['eval_total_mse'][-1]:.6f} "
                f"train_masked={curves['train_masked_mse'][-1]:.6f} "
                f"train_obs={curves['train_observed_mse'][-1]:.6f} "
                f"eval_masked={curves['eval_masked_mse'][-1]:.6f} "
                f"masked_abs={curves['train_masked_abs_pred_mean'][-1]:.6f}"
            )

        if (done % max(1, debug_every) == 0 or done == steps) and b_ev is not None and pred_ev is not None:
            idx = 0
            _pretty_print_example(
                pred=pred_ev[idx].detach().cpu().numpy(),
                target=b_ev.target[idx].detach().cpu().numpy(),
                observed_mask=b_ev.observed_mask[idx].detach().cpu().numpy(),
            )

    results = _payload(completed_steps=steps, live=False)
    (out_dir / "curves.json").write_text(json.dumps(results, indent=2))
    if live_curves_every > 0:
        _atomic_write_json(out_dir / "curves_live.json", results)
    render_plots_from_results(results, out_dir, log_y=True)
    print(f"Wrote curves and plots to {out_dir}")
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="BERT-style toy matrix completion.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--steps", type=int, default=NUM_STEPS)
    p.add_argument("--N", type=int, default=N_ROWS)
    p.add_argument("--M", type=int, default=N_COLS)
    p.add_argument("--rank", type=int, default=RANK)
    p.add_argument("--mask-prob", type=float, default=MASK_PROB)
    p.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    p.add_argument("--min-vocab", type=float, default=MIN_VOCAB)
    p.add_argument("--max-vocab", type=float, default=MAX_VOCAB)
    p.add_argument("--prec", type=int, default=PREC)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    p.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    p.add_argument("--num-heads", type=int, default=NUM_HEADS)
    p.add_argument("--intermediate-size", type=int, default=INTERMEDIATE_SIZE)
    p.add_argument(
        "--positional-scheme",
        type=str,
        default=POSITIONAL_SCHEME,
        choices=("flat", "rowcol_concat"),
        help="Position embedding scheme: flat absolute index, or concat(row_abs, col_abs).",
    )
    p.add_argument(
        "--continuous-tokenization",
        action="store_true",
        help=(
            "If set, bypass token lookup and inject scalar input directly into hidden state: "
            "h[...,0]=value, h[...,1:]=0."
        ),
    )
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--log-every", type=int, default=LOG_EVERY)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--debug-every", type=int, default=DEBUG_EVERY)
    p.add_argument("--live-curves-every", type=int, default=LIVE_CURVES_EVERY)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--replot", type=str, default=None)
    args = p.parse_args()

    if args.replot:
        src = Path(args.replot)
        results = json.loads(src.read_text())
        out = Path(args.out_dir) if args.out_dir else src.parent
        render_plots_from_results(results, out, log_y=True)
        print(f"Wrote plots to {out}")
        return

    out = Path(args.out_dir) if args.out_dir else OUT_DIR
    run_experiment(
        out_dir=out,
        seed=args.seed,
        steps=args.steps,
        n_rows=args.N,
        n_cols=args.M,
        rank=args.rank,
        mask_prob=args.mask_prob,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        min_vocab=args.min_vocab,
        max_vocab=args.max_vocab,
        prec=args.prec,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        positional_scheme=args.positional_scheme,
        continuous_tokenization=args.continuous_tokenization,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        eval_every=args.eval_every,
        debug_every=args.debug_every,
        live_curves_every=args.live_curves_every,
    )


if __name__ == "__main__":
    main()

