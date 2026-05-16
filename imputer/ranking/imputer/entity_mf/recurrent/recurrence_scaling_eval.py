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
            max_item=None,
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
    trained_r = rows[0]["trained_num_recurrence"] if rows else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rs, lls, "o-", color="#1f77b4", linewidth=2, markersize=8, alpha=0.85)
    if trained_r is not None and trained_r in rs:
        i = rs.index(trained_r)
        ax.scatter([trained_r], [lls[i]], color="#d62728", s=120, zorder=5, label=f"trained r={trained_r}")
    ax.set_xlabel("num_recurrence at eval")
    ax.set_ylabel("Test missing log loss (nats)")
    ax.set_title(
        f"Recurrence scaling — {summary.get('trained_config', '')} "
        f"({summary.get('checkpoint', '')} weights)"
    )
    ax.grid(True, alpha=0.3)
    if trained_r is not None:
        ax.legend(loc="best")
    fig.tight_layout()
    png = out_dir / "recurrence_scaling.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Wrote {png}")
    return png


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep num_recurrence at eval time for a trained Recurrent Marformer."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="last")
    parser.add_argument(
        "--recurrences",
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated recurrence counts to evaluate.",
    )
    parser.add_argument("--out-dir", default=None, help="Default: <run-dir>/RECURRENCE_SCALING")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    recurrences = _parse_recurrences(args.recurrences)
    out_dir = Path(args.out_dir) if args.out_dir else None

    summary = evaluate_recurrence_sweep(
        run_dir,
        checkpoint=args.checkpoint,
        recurrences=recurrences,
        device=device,
        out_dir=out_dir,
    )
    if not args.no_plot:
        plot_dir = out_dir or (run_dir / "RECURRENCE_SCALING")
        plot_recurrence_scaling(summary, plot_dir)


if __name__ == "__main__":
    main()
