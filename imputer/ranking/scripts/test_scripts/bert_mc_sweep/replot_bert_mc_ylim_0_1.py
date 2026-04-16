#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


def _load_results(run_dir: Path) -> Dict[str, Any]:
    live = run_dir / "curves_live.json"
    final = run_dir / "curves.json"
    if final.exists():
        return json.loads(final.read_text())
    if live.exists():
        return json.loads(live.read_text())
    raise FileNotFoundError(f"No curves.json or curves_live.json in {run_dir}")


def _render(results: Dict[str, Any], out_dir: Path, *, ylim_top: float = 1.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = (
        ("train_total_mse", "Train Total MSE", "bert_mc_train_total_mse.png"),
        ("eval_total_mse", "Eval Total MSE", "bert_mc_eval_total_mse.png"),
        ("train_masked_mse", "Train Masked-entry MSE", "bert_mc_train_masked_mse.png"),
        ("eval_masked_mse", "Eval Masked-entry MSE", "bert_mc_eval_masked_mse.png"),
    )
    for k, _t, _f in keys:
        if k not in results:
            raise KeyError(f"Missing {k} in results")

    n = len(results["train_total_mse"])
    x = np.arange(1, n + 1)

    for key, title, fname in keys:
        y = np.asarray(results[key], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8.6, 5.2))
        ax.plot(x, y, linewidth=2.0)
        ax.set_xlabel("step")
        ax.set_ylabel("MSE")
        ax.set_title(title + " (y in [0,1])")
        ax.set_ylim(bottom=0.0, top=float(ylim_top))
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Replot BERT MC curves with y-axis [0,1].")
    p.add_argument("run_dir", type=str, nargs="+", help="Run directory containing curves.json or curves_live.json")
    p.add_argument("--ylim-top", type=float, default=1.0)
    args = p.parse_args()

    for rd in args.run_dir:
        d = Path(rd).resolve()
        results = _load_results(d)
        _render(results, d, ylim_top=args.ylim_top)
        print(f"Wrote y-limited plots to {d}")


if __name__ == "__main__":
    main()

