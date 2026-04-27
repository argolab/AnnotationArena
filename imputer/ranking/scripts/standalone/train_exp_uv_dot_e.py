#!/usr/bin/env python3
"""
Standalone tensor-completion trainer for:

    z_ijk = exp(u_i + v_j) · e_k

where u_i, v_j, e_k are learned embeddings.

Expected input:
  --data-dir <dir containing data_bundle.json>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class RatingRow:
    i: int
    j: int
    k: int
    y: float
    instance: str


def _load_rows(bundle_path: Path, include_missing_rows: bool = False) -> Tuple[List[RatingRow], int]:
    with open(bundle_path, "r") as f:
        bundle = json.load(f)
    rows: List[RatingRow] = []
    dropped_non_finite = 0
    all_rows = list(bundle.get("observed_ratings", []))
    if include_missing_rows:
        all_rows += list(bundle.get("missing_ratings", []))
    for r in all_rows:
        y_raw = r.get("value", None)
        if y_raw is None:
            dropped_non_finite += 1
            continue
        y_val = float(y_raw)
        if not math.isfinite(y_val):
            dropped_non_finite += 1
            continue
        rows.append(
            RatingRow(
                i=int(r["attribute"]) - 1,
                j=int(r["annotator"]) - 1,
                k=int(r["item"]) - 1,
                y=y_val,
                instance=str(r["instance"]),
            )
        )
    return rows, dropped_non_finite


def _split_rows(rows: List[RatingRow]) -> Dict[str, List[RatingRow]]:
    out: Dict[str, List[RatingRow]] = {"train": [], "val": [], "test": []}
    for r in rows:
        if r.instance in out:
            out[r.instance].append(r)
    return out


def _max_index(rows: List[RatingRow], attr: str) -> int:
    return max(getattr(r, attr) for r in rows) + 1


def _to_tensors(rows: List[RatingRow], device: torch.device) -> Tuple[torch.Tensor, ...]:
    i = torch.tensor([r.i for r in rows], dtype=torch.long, device=device)
    j = torch.tensor([r.j for r in rows], dtype=torch.long, device=device)
    k = torch.tensor([r.k for r in rows], dtype=torch.long, device=device)
    y = torch.tensor([r.y for r in rows], dtype=torch.float32, device=device)
    return i, j, k, y


def _target_stats(rows: List[RatingRow]) -> Dict[str, float]:
    if not rows:
        return {"n": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    y = torch.tensor([r.y for r in rows], dtype=torch.float32)
    return {
        "n": float(y.numel()),
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "min": float(y.min().item()),
        "max": float(y.max().item()),
    }


def _entity_overlap(train_rows: List[RatingRow], val_rows: List[RatingRow]) -> Dict[str, int]:
    train_i = {r.i for r in train_rows}
    train_j = {r.j for r in train_rows}
    train_k = {r.k for r in train_rows}
    val_i = {r.i for r in val_rows}
    val_j = {r.j for r in val_rows}
    val_k = {r.k for r in val_rows}
    return {
        "train_i": len(train_i),
        "train_j": len(train_j),
        "train_k": len(train_k),
        "val_i": len(val_i),
        "val_j": len(val_j),
        "val_k": len(val_k),
        "val_i_unseen": len(val_i - train_i),
        "val_j_unseen": len(val_j - train_j),
        "val_k_unseen": len(val_k - train_k),
    }


def _triple_overlap(train_rows: List[RatingRow], val_rows: List[RatingRow]) -> Dict[str, int]:
    train_keys = {(r.i, r.j, r.k) for r in train_rows}
    val_keys = {(r.i, r.j, r.k) for r in val_rows}
    inter = train_keys & val_keys
    return {
        "train_unique_triples": len(train_keys),
        "val_unique_triples": len(val_keys),
        "shared_triples": len(inter),
        "val_only_triples": len(val_keys - train_keys),
    }


def _mean_baseline_mse(rows: List[RatingRow], train_mean: float) -> float:
    if not rows:
        return 0.0
    y = torch.tensor([r.y for r in rows], dtype=torch.float32)
    pred = torch.full_like(y, fill_value=train_mean)
    return float(F.mse_loss(pred, y).item())


def _load_bundle(bundle_path: Path) -> Dict:
    with open(bundle_path, "r") as f:
        return json.load(f)


def _print_bundle_label_diagnostics(bundle: Dict, *, j_dim: int) -> None:
    """Cheap checks for common synthetic-bundle inconsistencies."""
    obs = bundle.get("observed_ratings") or []
    if not obs:
        return
    bs = bundle.get("base_scores")
    if bs is None:
        print("Bundle diagnostics: base_scores missing; skipping oracle checks.")
        return

    arr = np.asarray(bs, dtype=np.float64)
    ys = np.array([float(r.get("value", 0.0)) for r in obs], dtype=np.float64)
    zero_frac = float(np.mean(np.abs(ys) <= 1e-12))

    sq = []
    sq_nz = []
    sq_z = []
    for r in obs:
        i = int(r["attribute"]) - 1
        j = int(r["annotator"]) - 1
        k = int(r["item"]) - 1
        y = float(r.get("value", 0.0))
        pred = float(arr[i * j_dim + j, k])
        e = (pred - y) ** 2
        sq.append(e)
        if abs(y) <= 1e-12:
            sq_z.append(e)
        else:
            sq_nz.append(e)

    mse_all = float(np.mean(sq)) if sq else float("nan")
    mse_nz = float(np.mean(sq_nz)) if sq_nz else float("nan")
    mse_z = float(np.mean(sq_z)) if sq_z else float("nan")
    print(
        "Bundle diagnostics | observed_zero_frac="
        f"{zero_frac:.3f} oracle_mse_all={mse_all:.6g} "
        f"oracle_mse_y!=0={mse_nz:.6g} oracle_mse_y==0={mse_z:.6g}"
    )
    if zero_frac > 0.05 and (not np.isnan(mse_z)) and mse_z > 1.0:
        print(
            "WARNING: Many observed labels are exactly 0 but disagree strongly with base_scores. "
            "This usually means stale Stan compilation (rerun generate_data.py with --force-stan-recompile) "
            "or a generator/extraction mismatch."
        )


@torch.no_grad()
def _initialize_from_bundle_ground_truth(
    model: nn.Module,
    bundle: Dict,
    num_i: int,
    num_j: int,
    num_k: int,
    gate_mode: str = "sum",
) -> None:
    """
    Initialize e from bundle['embeddings'] and approximate u,v from
    bundle['base_scores'] via:
      base_scores[i,j,:] ~= g_ij @ e^T, with g_ij = exp(u_i + v_j) > 0.
    """
    emb = bundle.get("embeddings")
    bs = bundle.get("base_scores")
    if emb is None or bs is None:
        raise ValueError("Bundle must contain 'embeddings' and 'base_scores' for --init-from-bundle-ground-truth.")

    e_all = np.asarray(emb, dtype=np.float64)  # [K_total, D]
    s_all = np.asarray(bs, dtype=np.float64)   # [I*J, K_total]
    if e_all.ndim != 2 or s_all.ndim != 2:
        raise ValueError("Invalid shapes for embeddings/base_scores.")
    if s_all.shape[0] < num_i * num_j:
        raise ValueError(f"base_scores rows ({s_all.shape[0]}) < num_i*num_j ({num_i*num_j})")
    if e_all.shape[0] < num_k or s_all.shape[1] < num_k:
        raise ValueError("embeddings/base_scores do not cover requested num_k")

    e = e_all[:num_k, : model.e.weight.shape[1]]  # [K, D]
    s = s_all[: num_i * num_j, :num_k]            # [I*J, K]

    # Solve g in least-squares sense: s ~= g @ e^T  => g ~= s @ pinv(e^T)
    e_t_pinv = np.linalg.pinv(e.T)                 # [K, D]
    g = s @ e_t_pinv                               # [I*J, D]
    g = np.clip(g, 1e-8, None)
    l = np.log(g).reshape(num_i, num_j, -1)       # additive model target

    if gate_mode == "sum":
        # Two-way additive fit per dim: l_ij ~= u_i + v_j
        grand = l.mean(axis=(0, 1), keepdims=True)     # [1,1,D]
        row_mean = l.mean(axis=1, keepdims=True)       # [I,1,D]
        col_mean = l.mean(axis=0, keepdims=True)       # [1,J,D]
        u = (row_mean - grand).reshape(num_i, -1)      # [I,D]
        v = col_mean.reshape(num_j, -1)                # [J,D]
    elif gate_mode == "product":
        # Rank-1 multiplicative fit per dim: l_ij ~= u_i * v_j
        u = np.zeros((num_i, l.shape[2]), dtype=np.float64)
        v = np.zeros((num_j, l.shape[2]), dtype=np.float64)
        for d in range(l.shape[2]):
            m = l[:, :, d]
            uu, ss, vv_t = np.linalg.svd(m, full_matrices=False)
            s0 = max(float(ss[0]), 1e-12)
            a = uu[:, 0] * np.sqrt(s0)
            b = vv_t[0, :] * np.sqrt(s0)
            u[:, d] = a
            v[:, d] = b
    else:
        raise ValueError(f"Unknown gate_mode={gate_mode!r}; expected 'sum' or 'product'.")

    u_t = torch.tensor(u, dtype=model.u.weight.dtype, device=model.u.weight.device)
    v_t = torch.tensor(v, dtype=model.v.weight.dtype, device=model.v.weight.device)
    e_t = torch.tensor(e, dtype=model.e.weight.dtype, device=model.e.weight.device)

    if u_t.shape != model.u.weight.shape or v_t.shape != model.v.weight.shape or e_t.shape != model.e.weight.shape:
        raise ValueError(
            f"Shape mismatch in GT init: "
            f"u {u_t.shape} vs {model.u.weight.shape}, "
            f"v {v_t.shape} vs {model.v.weight.shape}, "
            f"e {e_t.shape} vs {model.e.weight.shape}"
        )

    model.u.weight.copy_(u_t)
    model.v.weight.copy_(v_t)
    model.e.weight.copy_(e_t)


@torch.no_grad()
def _count_rows_deviated_from_init(
    emb: nn.Embedding,
    init_weight: torch.Tensor,
    tol: float = 1e-8,
) -> int:
    """Count how many embedding rows moved from initialization by > tol in L2 norm."""
    if emb.weight.shape != init_weight.shape:
        raise ValueError("Embedding and init_weight must have same shape.")
    delta = emb.weight.detach() - init_weight.to(emb.weight.device, emb.weight.dtype)
    row_l2 = torch.linalg.vector_norm(delta, ord=2, dim=1)
    return int((row_l2 > tol).sum().item())


class ExpUVDotE(nn.Module):
    def __init__(
        self,
        num_i: int,
        num_j: int,
        num_k: int,
        dim: int,
        init_std_e: float = 1e-3,
        init_std_uv: float = 1e-2,
        gate_mode: str = "sum",
        strict_formula: bool = False,
    ) -> None:
        super().__init__()
        self.u = nn.Embedding(num_i, dim)
        self.v = nn.Embedding(num_j, dim)
        self.e = nn.Embedding(num_k, dim)
        self.gate_mode = gate_mode
        self.strict_formula = bool(strict_formula)
        # Sum gate: u,v at 0 -> exp(u+v)=1 (stable).
        # Product gate: u=v=0 makes ∂pred/∂u ∝ v and ∂pred/∂v ∝ u both vanish; use
        # independent small normals so u_d and v_d are almost never simultaneously zero.
        if gate_mode == "product":
            nn.init.normal_(self.u.weight, mean=0.0, std=init_std_uv)
            nn.init.normal_(self.v.weight, mean=0.0, std=init_std_uv)
        else:
            nn.init.zeros_(self.u.weight)
            nn.init.zeros_(self.v.weight)
        nn.init.normal_(self.e.weight, mean=0.0, std=init_std_e)

    def forward(self, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "sum":
            log_gate = self.u(i) + self.v(j)
        elif self.gate_mode == "product":
            log_gate = self.u(i) * self.v(j)
        else:
            raise ValueError(f"Unsupported gate_mode={self.gate_mode!r}")

        if self.strict_formula:
            gate = torch.exp(log_gate)
            e_vec = self.e(k)
            return torch.sum(gate * e_vec, dim=-1)

        # Stabilized mode.
        gate = torch.exp(torch.clamp(log_gate, min=-8.0, max=8.0))
        e_vec = torch.tanh(self.e(k))
        return torch.sum(gate * e_vec, dim=-1) / (gate.shape[-1] ** 0.5)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    rows: List[RatingRow],
    device: torch.device,
) -> Dict[str, float]:
    if not rows:
        return {"mse": 0.0, "mae": 0.0, "n": 0}
    i, j, k, y = _to_tensors(rows, device=device)
    pred = model(i, j, k)
    mse = F.mse_loss(pred, y).item()
    mae = F.l1_loss(pred, y).item()
    return {"mse": float(mse), "mae": float(mae), "n": int(y.numel())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone exp(u+v)·e tensor model.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing data_bundle.json.")
    parser.add_argument("--output-dir", type=str, default="RESULTS/standalone_exp_uv_dot_e")
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-std-e", type=float, default=1e-3, help="Stddev for initial item embeddings e_k.")
    parser.add_argument(
        "--init-std-uv",
        type=float,
        default=1e-2,
        help="For --gate-mode product only: stddev for independent Normal init of u and v (avoids u=v=0 dead zone).",
    )
    parser.add_argument(
        "--swap-train-val",
        action="store_true",
        help="Ad hoc diagnostic: train on original val split and evaluate on original train split.",
    )
    parser.add_argument(
        "--override-train-val-ratio",
        type=float,
        default=None,
        help=(
            "Optional override: reshuffle current train+val rows and split by this ratio "
            "(e.g. 0.8 => 80%% train, 20%% val). Test split is unchanged."
        ),
    )
    parser.add_argument(
        "--init-from-bundle-ground-truth",
        action="store_true",
        help="Initialize model from bundle embeddings/base_scores and print pre-train losses.",
    )
    parser.add_argument(
        "--gate-mode",
        type=str,
        default="sum",
        choices=["sum", "product"],
        help="Gate composition for exp(): 'sum' uses u+v; 'product' uses u*v (matches tensor_nobin generator).",
    )
    parser.add_argument(
        "--strict-formula",
        action="store_true",
        help="Use exact formula without stabilization: pred = dot(exp(log_gate), e_k).",
    )
    parser.add_argument(
        "--include-missing-rows",
        action="store_true",
        help=(
            "Include missing_ratings rows as supervised targets. "
            "Default is observed-only, which is correct for synthetic bundles where "
            "missing rows store placeholder values."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    bundle_path = data_dir / "data_bundle.json"
    bundle_dict = _load_bundle(bundle_path)
    rows, dropped_non_finite = _load_rows(bundle_path, include_missing_rows=bool(args.include_missing_rows))
    if dropped_non_finite > 0:
        print(f"Dropped {dropped_non_finite} rows with non-finite targets.")
    if args.include_missing_rows:
        print("Warning: --include-missing-rows is enabled; ensure missing rows contain real labels.")
    split = _split_rows(rows)
    if args.swap_train_val:
        split["train"], split["val"] = split["val"], split["train"]
        print(
            "Split override enabled (--swap-train-val): "
            "training on original val rows; validating on original train rows."
        )
    if args.override_train_val_ratio is not None:
        ratio = float(args.override_train_val_ratio)
        if not (0.0 < ratio < 1.0):
            raise ValueError("--override-train-val-ratio must be in (0, 1).")
        pool = list(split["train"]) + list(split["val"])
        if len(pool) < 2:
            raise ValueError("Need at least 2 rows across train+val for ratio override.")
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        n_train_new = int(len(pool) * ratio)
        n_train_new = max(1, min(n_train_new, len(pool) - 1))
        split["train"] = pool[:n_train_new]
        split["val"] = pool[n_train_new:]
        print(
            "Split override enabled (--override-train-val-ratio): "
            f"ratio={ratio:.3f}, train={len(split['train'])}, val={len(split['val'])}, "
            f"test={len(split['test'])} (test unchanged)."
        )

    train_rows = split["train"]
    if not train_rows:
        raise ValueError("No training rows found in data_bundle.json")
    val_rows = split["val"]

    num_i = _max_index(rows, "i")
    num_j = _max_index(rows, "j")
    num_k = _max_index(rows, "k")
    _print_bundle_label_diagnostics(bundle_dict, j_dim=num_j)

    model = ExpUVDotE(
        num_i=num_i,
        num_j=num_j,
        num_k=num_k,
        dim=args.embed_dim,
        init_std_e=args.init_std_e,
        init_std_uv=float(args.init_std_uv),
        gate_mode=args.gate_mode,
        strict_formula=bool(args.strict_formula),
    ).to(device)
    init_u = model.u.weight.detach().clone()
    init_v = model.v.weight.detach().clone()
    init_e = model.e.weight.detach().clone()
    if args.init_from_bundle_ground_truth:
        _initialize_from_bundle_ground_truth(
            model,
            bundle_dict,
            num_i=num_i,
            num_j=num_j,
            num_k=num_k,
            gate_mode=args.gate_mode,
        )
        pre_train_eval = evaluate(model, split["train"], device=device)
        pre_val_eval = evaluate(model, split["val"], device=device)
        pre_test_eval = evaluate(model, split["test"], device=device)
        print(
            "Pre-train loss after GT init | "
            f"train_mse={pre_train_eval['mse']:.6f} "
            f"val_mse={pre_val_eval['mse']:.6f} "
            f"test_mse={pre_test_eval['mse']:.6f}"
        )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ti, tj, tk, ty = _to_tensors(train_rows, device=device)
    train_stats = _target_stats(train_rows)
    val_stats = _target_stats(val_rows)
    test_stats = _target_stats(split["test"])
    overlap = _entity_overlap(train_rows, val_rows)
    triple_overlap = _triple_overlap(train_rows, val_rows)
    print(
        "Train->Val entity coverage | "
        f"val unseen: i={overlap['val_i_unseen']}/{overlap['val_i']} "
        f"j={overlap['val_j_unseen']}/{overlap['val_j']} "
        f"k={overlap['val_k_unseen']}/{overlap['val_k']}"
    )
    print(
        "Train<->Val triple overlap | "
        f"shared={triple_overlap['shared_triples']} "
        f"val_only={triple_overlap['val_only_triples']} "
        f"(train_unique={triple_overlap['train_unique_triples']}, "
        f"val_unique={triple_overlap['val_unique_triples']})"
    )
    print(
        "Target stats | "
        f"train(n={int(train_stats['n'])}, mean={train_stats['mean']:.6g}, std={train_stats['std']:.6g}, min={train_stats['min']:.6g}, max={train_stats['max']:.6g}) | "
        f"val(n={int(val_stats['n'])}, mean={val_stats['mean']:.6g}, std={val_stats['std']:.6g}, min={val_stats['min']:.6g}, max={val_stats['max']:.6g}) | "
        f"test(n={int(test_stats['n'])}, mean={test_stats['mean']:.6g}, std={test_stats['std']:.6g}, min={test_stats['min']:.6g}, max={test_stats['max']:.6g})"
    )

    y_mean_t = ty.mean()
    y_std_t = ty.std(unbiased=False).clamp_min(1e-6)
    y_mean = float(y_mean_t.item())
    y_std = float(y_std_t.item())
    print(f"Target normalization: mean={y_mean:.6g}, std={y_std:.6g} (not used in loss)")
    train_var = max(y_std * y_std, 1e-12)
    base_train_mse = _mean_baseline_mse(train_rows, y_mean)
    base_val_mse = _mean_baseline_mse(split["val"], y_mean)
    base_test_mse = _mean_baseline_mse(split["test"], y_mean)
    print(
        "Mean baseline MSE (predict train mean) | "
        f"train={base_train_mse:.6g} val={base_val_mse:.6g} test={base_test_mse:.6g}"
    )

    n = ty.numel()
    index = torch.arange(n, device=device)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = index[torch.randperm(n, device=device)]
        train_loss_sum = 0.0
        train_count = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred = model(ti[idx], tj[idx], tk[idx])
            loss = F.mse_loss(pred, ty[idx])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered. Try smaller --lr.")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optim.step()
            train_loss_sum += float(loss.item()) * int(idx.numel())
            train_count += int(idx.numel())

        train_mse = train_loss_sum / max(train_count, 1)
        train_eval = evaluate(model, train_rows, device=device)
        val_eval = evaluate(model, split["val"], device=device)
        test_eval = evaluate(model, split["test"], device=device)

        record = {
            "epoch": epoch,
            "train_mse_batch": float(train_mse),
            "train_mse_full": train_eval["mse"],
            "val_mse": val_eval["mse"],
            "test_mse": test_eval["mse"],
            "train_nmse": float(train_eval["mse"] / train_var),
            "val_nmse": float(val_eval["mse"] / train_var),
            "test_nmse": float(test_eval["mse"] / train_var),
            "train_mae": train_eval["mae"],
            "val_mae": val_eval["mae"],
            "test_mae": test_eval["mae"],
            "u_rows_deviated": _count_rows_deviated_from_init(model.u, init_u),
            "v_rows_deviated": _count_rows_deviated_from_init(model.v, init_v),
            "e_rows_deviated": _count_rows_deviated_from_init(model.e, init_e),
        }
        history.append(record)
        print(
            f"[{epoch:04d}] train_mse={record['train_mse_full']:.6f} "
            f"val_mse={record['val_mse']:.6f} test_mse={record['test_mse']:.6f} "
            f"| train_nmse={record['train_nmse']:.4f} val_nmse={record['val_nmse']:.4f} "
            f"test_nmse={record['test_nmse']:.4f} "
            f"| moved_rows: u={record['u_rows_deviated']}/{num_i} "
            f"v={record['v_rows_deviated']}/{num_j} e={record['e_rows_deviated']}/{num_k}"
        )

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_i": num_i,
            "num_j": num_j,
            "num_k": num_k,
            "embed_dim": args.embed_dim,
            "formula": "z_ijk = dot(exp(u_i + v_j), e_k)",
        },
        ckpt_path,
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(
            {
                "train": evaluate(model, train_rows, device=device),
                "val": evaluate(model, split["val"], device=device),
                "test": evaluate(model, split["test"], device=device),
                "baseline_mse_predict_train_mean": {
                    "train": base_train_mse,
                    "val": base_val_mse,
                    "test": base_test_mse,
                },
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
