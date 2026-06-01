#!/usr/bin/env python3
"""
LLM Rubric: structured NB factor ablations vs train size (+ unigram IJ, IJK baselines).

Evaluates test-missing mean NLL and RMSE for each bundle under DATA/LLM_RUBRIC
(LLMRubric_225_25_9_{size}/data_bundle.json), matching the sizing used in
plot_llm_rubric_cpm_with_structured_baselines.py.

Structured variants (IJK slots always on; pairwise factors toggled):
  - snb_full          : attr_pair + CHANGEJ (default structured NB)
  - snb_no_change_j   : without CHANGEJ
  - snb_no_attr       : without attr-pair P_{i',i}
  - snb_ijk_only      : without both pairwise factors (= IJK slots only)
  - unigram_ij        : pooled P(y|i,j)
  - ijk               : naive Bayes IJK

Run from imputer/ranking:

  python scripts/utils/run_llm_rubric_snb_ablations.py

  python scripts/utils/run_llm_rubric_snb_ablations.py --plot

  python scripts/utils/run_llm_rubric_snb_ablations.py --sizes 10,50,175 --out RESULTS/llm_rubric_snb_ablations.json

Plot from existing JSON (no re-fit):

  python scripts/utils/run_llm_rubric_snb_ablations.py \\
      --plot-from-json RESULTS/llm_rubric_snb_ablations.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

_RANKING_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RANKING_ROOT / "BASELINES"))

from structured_baselines.cli_defaults import DEFAULT_IJK_ALPHA, DEFAULT_SNB_ALPHA, DEFAULT_UNIGRAM_ALPHA
from structured_baselines.dataset_adapter import build_test_examples, bundle_dims, load_bundle_dict
from structured_baselines.naive_bayes_ijk import NaiveBayesIJK
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes
from structured_baselines.plate_graph_factorized import StructuredFactorMask
from structured_baselines.unigram_pooled import PooledUnigramIJ

BUNDLE_DIR_RE = re.compile(r"^LLMRubric_225_25_9_(\d+)$")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    kind: str  # "structured" | "unigram_ij" | "ijk"
    factor_mask: StructuredFactorMask | None = None


def _all_variants() -> list[ModelSpec]:
    T = True
    F = False
    return [
        ModelSpec("unigram_ij", "Unigram P(y|i,j)", "unigram_ij"),
        ModelSpec("ijk", "Naive Bayes IJK", "ijk"),
        ModelSpec(
            "snb_full",
            "Structured NB (full)",
            "structured",
            StructuredFactorMask(attr_pair=T, change_j=T),
        ),
        ModelSpec(
            "snb_no_change_j",
            "Structured NB (−CHANGEJ)",
            "structured",
            StructuredFactorMask(attr_pair=T, change_j=F),
        ),
        ModelSpec(
            "snb_no_attr",
            "Structured NB (−attr-pair)",
            "structured",
            StructuredFactorMask(attr_pair=F, change_j=T),
        ),
        ModelSpec(
            "snb_ijk_only",
            "Structured NB (IJK slots only)",
            "structured",
            StructuredFactorMask(attr_pair=F, change_j=F),
        ),
    ]


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


def _evaluate_variant(
    spec: ModelSpec,
    bundle: dict,
    bundle_path: Path,
    test_ex: list,
    *,
    snb_alpha: float,
    ijk_alpha: float,
    unigram_alpha: float,
) -> dict[str, float]:
    if spec.kind == "unigram_ij":
        model = PooledUnigramIJ.fit(bundle, alpha=unigram_alpha)
        ev = model.evaluate_split(bundle, "test")
        return {
            "accuracy": float(ev["accuracy"]),
            "mean_nll": float(ev["mean_nll"]),
            "rmse": float(ev["rmse"]),
            "n": float(ev["n"]),
        }
    if spec.kind == "ijk":
        model = NaiveBayesIJK.fit_from_bundle(bundle, alpha=ijk_alpha)
        ev = model.evaluate(test_ex)
        probs = model.predict_proba(test_ex)
        return {
            "accuracy": float(ev["accuracy"]),
            "mean_nll": float(ev["mean_nll"]),
            "rmse": _rmse_from_proba(test_ex, probs),
            "n": float(ev["n"]),
        }
    assert spec.kind == "structured" and spec.factor_mask is not None
    I, J, C = bundle_dims(bundle, bundle_path)
    K = max(
        int(r["item"])
        for r in (bundle.get("observed_ratings", []) + bundle.get("missing_ratings", []))
    )
    model = StructuredNaiveBayes.fit_from_bundle(
        bundle,
        num_attrs=I,
        num_classes=C,
        num_anns=J,
        num_items=K,
        alpha=snb_alpha,
        factor_mask=spec.factor_mask,
    )
    ev = model.evaluate(test_ex)
    probs = model.predict_proba(test_ex)
    return {
        "accuracy": float(ev["accuracy"]),
        "mean_nll": float(ev["mean_nll"]),
        "rmse": _rmse_from_proba(test_ex, probs),
        "n": float(ev["n"]),
        "factor_mask": {
            "attr_pair": spec.factor_mask.attr_pair,
            "change_j": spec.factor_mask.change_j,
        },
    }


# Omit from ablation plots (SNB variants only).
_PLOT_SKIP_KEYS = frozenset({"unigram_ij", "ijk"})


def _series_from_results(results: dict, metric: str) -> tuple[dict[str, list[tuple[int, float]]], dict[str, str]]:
    label_map: dict[str, str] = {
        k: v for k, v in results.get("variants", {}).items() if k not in _PLOT_SKIP_KEYS
    }
    series: dict[str, list[tuple[int, float]]] = {k: [] for k in label_map}
    by_size = results.get("by_size", {})
    for size_str in sorted(by_size.keys(), key=int):
        size = int(size_str)
        for key, metrics in by_size[size_str].items():
            if key in _PLOT_SKIP_KEYS:
                continue
            if key not in series:
                series[key] = []
                label_map.setdefault(key, key)
            if metric in metrics:
                series[key].append((size, float(metrics[metric])))
    return series, label_map


def plot_from_json(
    json_path: Path,
    *,
    output_logloss: Path,
    output_rmse: Path,
    output_accuracy: Path | None = None,
) -> None:
    """Plot curves from a saved ablation JSON (no re-fit)."""
    results = json.loads(json_path.read_text())
    nll_series, label_map = _series_from_results(results, "mean_nll")
    rmse_series, _ = _series_from_results(results, "rmse")
    _plot_curves(
        nll_series,
        ylabel="Test missing mean NLL",
        title="LLM Rubric: structured NB factor ablations (log loss)",
        output=output_logloss,
        label_map=label_map,
    )
    _plot_curves(
        rmse_series,
        ylabel="Test missing RMSE",
        title="LLM Rubric: structured NB factor ablations (RMSE)",
        output=output_rmse,
        label_map=label_map,
    )
    if output_accuracy is not None:
        acc_series, _ = _series_from_results(results, "accuracy")
        _plot_curves(
            acc_series,
            ylabel="Test missing accuracy",
            title="LLM Rubric: structured NB factor ablations (accuracy)",
            output=output_accuracy,
            label_map=label_map,
        )


_PLOT_MARKERS: dict[str, dict[str, str]] = {
    "unigram_ij": {"marker": "o", "linestyle": "-"},
    "ijk": {"marker": "s", "linestyle": "-"},
    "snb_full": {"marker": "D", "linestyle": "-"},
    "snb_no_change_j": {"marker": "v", "linestyle": "--"},
    "snb_no_attr": {"marker": "P", "linestyle": "-."},
    "snb_ijk_only": {"marker": ">", "linestyle": ":"},
}
_SNB_COLORS = {
    "snb_full": "#e7298a",
    "snb_no_change_j": "#fc8d62",
    "snb_no_attr": "#7570b3",
    "snb_ijk_only": "#666666",
}
_BASELINE_COLORS = {"unigram_ij": "#0b7285", "ijk": "#333333"}
_FALLBACK_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "*")
_LINE_ALPHA = 0.52
_MARKER_ALPHA = 0.62


def _color_for_key(key: str) -> str:
    if key in _BASELINE_COLORS:
        return _BASELINE_COLORS[key]
    return _SNB_COLORS.get(key, "#666666")


def _style_for_key(key: str, idx: int) -> dict[str, str]:
    mk = _PLOT_MARKERS.get(key, {})
    return {
        "color": _color_for_key(key),
        "marker": mk.get("marker", _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)]),
        "linestyle": mk.get("linestyle", "-"),
    }


def _plot_curves(
    series: dict[str, list[tuple[int, float]]],
    *,
    ylabel: str,
    title: str,
    output: Path,
    label_map: dict[str, str],
) -> None:
    # Draw ablated variants first, full SNB last so it sits on top when overlapping.
    draw_order = [
        "snb_ijk_only",
        "snb_no_attr",
        "snb_no_change_j",
        "snb_full",
    ]
    keys_sorted = [k for k in draw_order if k in series and series[k]]
    keys_sorted += sorted(k for k in series if k not in keys_sorted and series[k])

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    for idx, key in enumerate(keys_sorted):
        pts = series[key]
        if not pts:
            continue
        pts = sorted(pts, key=lambda x: x[0])
        st = _style_for_key(key, idx)
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=2.0,
            markersize=7,
            markeredgewidth=0.6,
            markeredgecolor="white",
            alpha=_LINE_ALPHA,
            label=label_map.get(key, key),
            zorder=idx + 1,
        )
        # Slightly more opaque markers on top of semi-transparent lines
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=st["color"],
            marker=st["marker"],
            linestyle="none",
            markersize=7,
            markeredgewidth=0.6,
            markeredgecolor="white",
            alpha=_MARKER_ALPHA,
            zorder=idx + 1,
        )

    ax.set_xlabel("Training items (+25 test items with observed LLM ratings)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    xs = sorted({p[0] for pts in series.values() for p in pts})
    if xs:
        ax.set_xticks(xs)
    ax.grid(alpha=0.3)
    ax.legend(
        fontsize=7.5,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.92,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM Rubric structured NB factor ablations")
    ap.add_argument("--data-root", type=Path, default=Path("DATA/LLM_RUBRIC"))
    ap.add_argument(
        "--sizes",
        type=str,
        default="",
        help="Comma-separated train sizes (default: all LLMRubric_225_25_9_* under data-root)",
    )
    ap.add_argument("--snb-alpha", type=float, default=DEFAULT_SNB_ALPHA)
    ap.add_argument("--ijk-alpha", type=float, default=DEFAULT_IJK_ALPHA)
    ap.add_argument("--unigram-alpha", type=float, default=DEFAULT_UNIGRAM_ALPHA)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("RESULTS/llm_rubric_snb_ablations.json"),
    )
    ap.add_argument("--plot", action="store_true", help="Write log-loss and RMSE curve PNGs")
    ap.add_argument(
        "--plot-from-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Plot from saved JSON only; skip fitting",
    )
    ap.add_argument(
        "--output-logloss",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_snb_ablations_log_loss.png"),
    )
    ap.add_argument(
        "--output-rmse",
        type=Path,
        default=Path("PLOTS/TALK/LLM_RUBRIC/llm_rubric_snb_ablations_rmse.png"),
    )
    ap.add_argument(
        "--output-accuracy",
        type=Path,
        default=None,
        help="Optional third plot for accuracy",
    )
    args = ap.parse_args()

    if args.plot_from_json is not None:
        plot_from_json(
            args.plot_from_json,
            output_logloss=args.output_logloss,
            output_rmse=args.output_rmse,
            output_accuracy=args.output_accuracy,
        )
        return

    if args.sizes.strip():
        sizes = sorted(int(s.strip()) for s in args.sizes.split(",") if s.strip())
    else:
        sizes = _discover_sizes(args.data_root)
    if not sizes:
        raise SystemExit(f"No bundles found under {args.data_root}")

    variants = _all_variants()
    label_map = {v.key: v.label for v in variants}

    results: dict = {
        "data_root": str(args.data_root),
        "sizes": sizes,
        "snb_alpha": args.snb_alpha,
        "ijk_alpha": args.ijk_alpha,
        "unigram_alpha": args.unigram_alpha,
        "variants": {v.key: v.label for v in variants},
        "by_size": {},
    }

    nll_series: dict[str, list[tuple[int, float]]] = {v.key: [] for v in variants}
    rmse_series: dict[str, list[tuple[int, float]]] = {v.key: [] for v in variants}

    for size in sizes:
        bundle_path = args.data_root / f"LLMRubric_225_25_9_{size}" / "data_bundle.json"
        if not bundle_path.exists():
            print(f"[skip] missing {bundle_path}")
            continue
        print(f"=== train size {size} ===")
        bundle = load_bundle_dict(bundle_path)
        test_ex = build_test_examples(bundle)
        size_out: dict[str, dict] = {}

        for spec in variants:
            print(f"  {spec.key} …", flush=True)
            metrics = _evaluate_variant(
                spec,
                bundle,
                bundle_path,
                test_ex,
                snb_alpha=args.snb_alpha,
                ijk_alpha=args.ijk_alpha,
                unigram_alpha=args.unigram_alpha,
            )
            size_out[spec.key] = metrics
            print(
                f"    nll={metrics['mean_nll']:.4f}  "
                f"acc={metrics['accuracy']:.4f}  rmse={metrics['rmse']:.4f}"
            )
            nll_series[spec.key].append((size, metrics["mean_nll"]))
            rmse_series[spec.key].append((size, metrics["rmse"]))

        results["by_size"][str(size)] = size_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {args.out}")

    if args.plot:
        _plot_curves(
            nll_series,
            ylabel="Test missing mean NLL",
            title="LLM Rubric: structured NB factor ablations (log loss)",
            output=args.output_logloss,
            label_map=label_map,
        )
        _plot_curves(
            rmse_series,
            ylabel="Test missing RMSE",
            title="LLM Rubric: structured NB factor ablations (RMSE)",
            output=args.output_rmse,
            label_map=label_map,
        )


if __name__ == "__main__":
    main()
