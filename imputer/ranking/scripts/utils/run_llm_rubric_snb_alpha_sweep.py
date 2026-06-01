#!/usr/bin/env python3
"""
LLM Rubric: sweep Laplace α for structured NB (attr-pair + CHANGEJ).

For each train size under DATA/LLM_RUBRIC, fits count tables once, then evaluates
test-missing NLL / RMSE / accuracy for each α. Writes JSON and optional curves
(train size on x-axis, one line per α).

Run from imputer/ranking:

  python scripts/utils/run_llm_rubric_snb_alpha_sweep.py

  python scripts/utils/run_llm_rubric_snb_alpha_sweep.py \\
      --alphas 0.5,1,2,5,10,20,50 \\
      --out RESULTS/llm_rubric_snb_alpha_sweep.json

  # Replot only:
  python scripts/utils/run_llm_rubric_snb_alpha_sweep.py \\
      --plot-from-json RESULTS/llm_rubric_snb_alpha_sweep.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RANKING_ROOT / "BASELINES"))

from structured_baselines.dataset_adapter import (
    build_test_examples,
    bundle_dims,
    load_bundle_dict,
    transductive_observed_cells,
)
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes
from structured_baselines.plate_graph_factorized import (
    StructuredFactorMask,
    accumulate_transductive_counts,
)

BUNDLE_DIR_RE = re.compile(r"^LLMRubric_225_25_9_(\d+)$")
DEFAULT_ALPHAS = "0.5,1,2,5,10,20,50"
FULL_MASK = StructuredFactorMask.all_on()
_LINE_ALPHA = 0.55
_MARKER_ALPHA = 0.65


def _parse_alphas(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        a = float(p)
        if a <= 0.0:
            raise ValueError(f"α must be positive, got {a!r}")
        out.append(a)
    return out


def _discover_sizes(data_root: Path) -> list[int]:
    sizes: list[int] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        m = BUNDLE_DIR_RE.match(p.name)
        if m and (p / "data_bundle.json").exists():
            sizes.append(int(m.group(1)))
    return sorted(sizes)


def _rmse_from_proba(examples, probs: np.ndarray) -> float:
    classes = np.arange(1, probs.shape[1] + 1, dtype=np.float64)
    pred = probs @ classes
    truth = np.array([ex.y + 1 for ex in examples], dtype=np.float64)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def _alpha_color(alpha: float, alphas: list[float]) -> str:
    """Light (low α) → dark (high α) on viridis."""
    if len(alphas) <= 1:
        return "#440154"
    idx = alphas.index(alpha) if alpha in alphas else 0
    t = idx / max(len(alphas) - 1, 1)
    return plt.cm.viridis(t)  # type: ignore[return-value]


def plot_sweep_json(
    results: dict,
    *,
    output_logloss: Path,
    output_rmse: Path,
    output_accuracy: Path | None = None,
) -> None:
    sizes = [int(s) for s in results["sizes"]]
    alphas = [float(a) for a in results["alphas"]]
    by_size = results["by_size"]

    for metric, ylabel, out_path in (
        ("mean_nll", "Test missing mean NLL", output_logloss),
        ("rmse", "Test missing RMSE", output_rmse),
        *(
            [("accuracy", "Test missing accuracy", output_accuracy)]
            if output_accuracy is not None
            else []
        ),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for alpha in alphas:
            pts: list[tuple[int, float]] = []
            for size in sizes:
                block = by_size[str(size)]
                metrics = block.get(f"{alpha:g}") or block.get(str(alpha))
                if metrics is not None and metric in metrics:
                    pts.append((size, float(metrics[metric])))
            if not pts:
                continue
            pts = sorted(pts)
            color = _alpha_color(alpha, alphas)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                markeredgewidth=0.5,
                markeredgecolor="white",
                alpha=_LINE_ALPHA,
                label=f"α={alpha:g}",
                zorder=1,
            )
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                marker="o",
                linestyle="none",
                markersize=6,
                markeredgewidth=0.5,
                markeredgecolor="white",
                alpha=_MARKER_ALPHA,
                zorder=2,
            )
        ax.set_xlabel("Training items (+25 test items with observed LLM ratings)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"LLM Rubric: structured NB full model — α sweep ({ylabel})")
        ax.set_xticks(sizes)
        ax.grid(alpha=0.3)
        ax.legend(
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            title="Laplace α",
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def run_sweep(
    data_root: Path,
    sizes: list[int],
    alphas: list[float],
    *,
    factor_mask: StructuredFactorMask = FULL_MASK,
) -> dict:
    results: dict = {
        "data_root": str(data_root),
        "sizes": sizes,
        "alphas": alphas,
        "factor_mask": {
            "attr_pair": factor_mask.attr_pair,
            "change_j": factor_mask.change_j,
        },
        "by_size": {},
    }

    for size in sizes:
        bundle_path = data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
        if not bundle_path.exists():
            print(f"[skip] missing {bundle_path}")
            continue
        print(f"=== train size {size} ===", flush=True)
        bundle = load_bundle_dict(bundle_path)
        I, J, C = bundle_dims(bundle, bundle_path)
        K = max(
            int(r["item"])
            for r in (bundle.get("observed_ratings", []) + bundle.get("missing_ratings", []))
        )
        cells = transductive_observed_cells(bundle)
        counts = accumulate_transductive_counts(
            cells,
            num_attrs=I,
            num_classes=C,
            num_anns=J,
            num_items=K,
            factor_mask=factor_mask,
        )
        test_ex = build_test_examples(bundle)
        size_out: dict[str, dict] = {}

        for alpha in alphas:
            snb = StructuredNaiveBayes(counts=counts, alpha=alpha, factor_mask=factor_mask)
            ev = snb.evaluate(test_ex)
            probs = snb.predict_proba(test_ex)
            size_out[f"{alpha:g}"] = {
                "accuracy": float(ev["accuracy"]),
                "mean_nll": float(ev["mean_nll"]),
                "rmse": _rmse_from_proba(test_ex, probs),
                "n": float(ev["n"]),
            }
            print(
                f"  α={alpha:g}  nll={ev['mean_nll']:.4f}  "
                f"acc={ev['accuracy']:.4f}  rmse={size_out[f'{alpha:g}']['rmse']:.4f}",
                flush=True,
            )
        results["by_size"][str(size)] = size_out

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM Rubric SNB α sweep (full structured model)")
    ap.add_argument("--data-root", type=Path, default=Path("DATA/LLM_RUBRIC"))
    ap.add_argument("--sizes", type=str, default="", help="Comma-separated sizes (default: all)")
    ap.add_argument("--alphas", type=str, default=DEFAULT_ALPHAS)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("RESULTS/llm_rubric_snb_alpha_sweep.json"),
    )
    ap.add_argument(
        "--plot-from-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Plot from saved JSON; skip sweep",
    )
    ap.add_argument(
        "--output-logloss",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_snb_alpha_sweep_log_loss.png"),
    )
    ap.add_argument(
        "--output-rmse",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_snb_alpha_sweep_rmse.png"),
    )
    ap.add_argument(
        "--output-accuracy",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_snb_alpha_sweep_accuracy.png"),
    )
    ap.add_argument("--no-plot", action="store_true", help="Skip plotting after sweep")
    args = ap.parse_args()

    if args.plot_from_json is not None:
        results = json.loads(args.plot_from_json.read_text())
        plot_sweep_json(
            results,
            output_logloss=args.output_logloss,
            output_rmse=args.output_rmse,
            output_accuracy=args.output_accuracy,
        )
        return

    alphas = _parse_alphas(args.alphas)
    if args.sizes.strip():
        sizes = sorted(int(s.strip()) for s in args.sizes.split(",") if s.strip())
    else:
        sizes = _discover_sizes(args.data_root)
    if not sizes:
        raise SystemExit(f"No bundles under {args.data_root}")

    results = run_sweep(args.data_root, sizes, alphas)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}")

    if not args.no_plot:
        plot_sweep_json(
            results,
            output_logloss=args.output_logloss,
            output_rmse=args.output_rmse,
            output_accuracy=args.output_accuracy,
        )


if __name__ == "__main__":
    main()
