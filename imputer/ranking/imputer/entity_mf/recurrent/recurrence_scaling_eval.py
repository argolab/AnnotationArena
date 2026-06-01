"""
Evaluate a trained Recurrent Marformer while varying num_recurrence at test time.

Example (p1c2r3c1 last checkpoint, sweep recurrence 1–8):
  python -m imputer.entity_mf.recurrent.recurrence_scaling_eval \\
    --run-dir RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD/DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_p1c2r3c1 \\
    --checkpoint last --recurrences 1,2,3,4,5,6,7,8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch

from .test import _compute_metrics, _find_checkpoint, _load_checkpoint, _reconstruct
from ..eval import evaluate_entity_marformer_split


def _parse_recurrences(s: str) -> List[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("empty recurrence list")
    return out


def evaluate_recurrence_sweep(
    run_dir: Path,
    *,
    checkpoint: str,
    recurrences: List[int],
    device: torch.device,
    out_dir: Path | None = None,
    max_item: int | None = None,
) -> Dict[str, Any]:
    out_dir = out_dir or (run_dir / "RECURRENCE_SCALING")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, eval_vars, train_cfg = _reconstruct(run_dir)
    ckpt_path = _find_checkpoint(run_dir / "checkpoints", checkpoint)
    _load_checkpoint(model, ckpt_path, device)
    model.to(device)

    trained_r = int(model.recurrent_config.num_recurrence)
    prelude = int(model.recurrent_config.prelude_depth)
    core = int(model.recurrent_config.num_core_layers)
    coda = int(model.recurrent_config.coda_depth)
    print(f"max_item={max_item}")

    rows: List[Dict[str, Any]] = []
    for r in recurrences:
        model.recurrent_config.num_recurrence = r
        actual = prelude + core * r + coda
        print(f"\n--- num_recurrence={r} (actual_depth={actual}) ---")
        result = evaluate_entity_marformer_split(
            model=model,
            split="test",
            variables=eval_vars,
            types=model.types,
            global_param_dim=model.global_param_dim,
            device=device,
            max_item=max_item,
        )
        metrics = _compute_metrics(result)
        miss = metrics.get("missing", {})
        ll = miss.get("log_loss")
        print(
            f"  missing log_loss={ll:.4f}  rmse={miss.get('rmse')}  n={miss.get('n')}"
            if ll is not None
            else "  missing → none"
        )
        rows.append(
            {
                "prelude_depth": prelude,
                "num_core_layers": core,
                "num_recurrence": r,
                "coda_depth": coda,
                "actual_depth": actual,
                "trained_num_recurrence": trained_r,
                "checkpoint": ckpt_path.name,
                "missing": miss,
                "observed": metrics.get("observed", {}),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "trained_config": f"p{prelude}c{core}r{trained_r}c{coda}",
        "recurrences": recurrences,
        "max_item": max_item,
        "eval_max_item": max_item,
        "train_max_item": train_cfg.get("max_item"),
        "eval_out_dir": str(out_dir),
        "results": rows,
    }
    json_path = out_dir / "recurrence_scaling.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {json_path}")
    return summary


def plot_recurrence_scaling(summary: Dict[str, Any], out_dir: Path) -> Path:
    rows = summary["results"]
    rs = [row["num_recurrence"] for row in rows]
    lls = [row["missing"]["log_loss"] for row in rows]
    rmses = [row["missing"]["rmse"] for row in rows]
    trained_r = rows[0]["trained_num_recurrence"] if rows else None
    cfg = summary.get("trained_config", "")
    ckpt = summary.get("checkpoint", "")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, ys, ylab, is_log_loss in zip(
        axes, (lls, rmses), ("log loss (nats)", "RMSE"), (True, False)
    ):
        ax.plot(rs, ys, "o-", color="#1f77b4", linewidth=2, markersize=8, alpha=0.85)
        if trained_r is not None and trained_r in rs:
            i = rs.index(trained_r)
            ax.scatter(
                [trained_r],
                [ys[i]],
                color="#d62728",
                s=120,
                zorder=5,
                label=f"trained r={trained_r}",
            )
        ax.set_xlabel("num_recurrence at eval")
        ax.set_ylabel(f"Test missing {ylab}")
        if is_log_loss:
            ax.set_ylim(0.3, 0.8)
        ax.grid(True, alpha=0.3)
        if trained_r is not None:
            ax.legend(loc="best")
    max_item = summary.get("max_item")
    if max_item is None:
        mi_label = "full graph (max_item=None)"
        mi_tag = "fullgraph"
    else:
        mi_label = f"max_item={max_item}"
        mi_tag = f"maxitem{max_item}"
    fig.suptitle(f"Recurrence scaling — {cfg} ({ckpt} weights, {mi_label})")
    fig.tight_layout()
    png = out_dir / f"recurrence_scaling_{mi_tag}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep num_recurrence at eval time for a trained Recurrent Marformer."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Weights checkpoint: latest (highest epoch periodic/best), best, or filename.",
    )
    parser.add_argument(
        "--recurrences",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated recurrence counts to evaluate.",
    )
    parser.add_argument("--out-dir", default=None, help="Default: <run-dir>/RECURRENCE_SCALING")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-item",
        type=int,
        default=None,
        help="Chunk eval by item count (default: train_config training.max_item).",
    )
    parser.add_argument(
        "--full-graph",
        action="store_true",
        help="Evaluate on full transductive graph (max_item=None).",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    recurrences = _parse_recurrences(args.recurrences)
    out_dir = Path(args.out_dir) if args.out_dir else None

    with open(run_dir / "train_config.json") as f:
        train_cfg = json.load(f).get("training", {})
    if args.full_graph:
        max_item: int | None = None
    elif args.max_item is not None:
        max_item = args.max_item
    else:
        max_item = train_cfg.get("max_item")

    summary = evaluate_recurrence_sweep(
        run_dir,
        checkpoint=args.checkpoint,
        recurrences=recurrences,
        device=device,
        out_dir=out_dir,
        max_item=max_item,
    )
    if not args.no_plot:
        plot_dir = out_dir or (run_dir / "RECURRENCE_SCALING")
        plot_recurrence_scaling(summary, plot_dir)


if __name__ == "__main__":
    main()
