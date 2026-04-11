from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass
class RunData:
    run_dir: Path
    history: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]]
    label: str


def load_training_history(run_dir: Path) -> List[Dict[str, Any]]:
    """Load training_history.json from a single run directory."""
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"No training_history.json found in {run_dir}")
    with history_path.open("r") as f:
        return json.load(f)


def load_train_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load train_config.json if present, else return None."""
    cfg_path = run_dir / "train_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def summarize_run_label(config: Optional[Dict[str, Any]], run_dir: Path) -> str:
    """Build a short human-readable label for a run."""
    if not config:
        return run_dir.name

    training = config.get("training", {})
    transductive = training.get("transductive_learning", False)
    max_item = training.get("max_item", None)
    epochs = training.get("epochs", None)

    parts: List[str] = []
    parts.append("T" if transductive else "NT")
    if max_item is not None:
        parts.append(f"max_item={max_item}")
    if epochs is not None:
        parts.append(f"epochs={epochs}")

    label_body = ", ".join(parts) if parts else run_dir.name
    return f"{run_dir.name} ({label_body})"


def extract_total_loss(history: Sequence[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    epochs: List[int] = []
    losses: List[float] = []
    for entry in history:
        if "epoch" not in entry or "total_loss" not in entry:
            continue
        epochs.append(int(entry["epoch"]))
        losses.append(float(entry["total_loss"]))
    return epochs, losses


def _get_nested_metric(
    entry: Dict[str, Any],
    split_key: str,
    status: str,
    type_name: str,
    metric_key: str,
) -> Optional[float]:
    """Safely read a nested metric like split.metrics[status][type_name][metric_key]."""
    split = entry.get(split_key)
    if not split:
        return None
    metrics = split.get("metrics") or {}
    by_status = metrics.get(status) or {}
    by_type = by_status.get(type_name) or {}
    val = by_type.get(metric_key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_status_type_metric(
    history: Sequence[Dict[str, Any]],
    split_key: str,
    status: str,
    type_name: str,
    metric_key: str,
) -> Tuple[List[int], List[float]]:
    """Extract a per-epoch series for a given (split, status, type, metric)."""
    epochs: List[int] = []
    values: List[float] = []
    for entry in history:
        if "epoch" not in entry:
            continue
        val = _get_nested_metric(entry, split_key, status, type_name, metric_key)
        if val is None:
            continue
        epochs.append(int(entry["epoch"]))
        values.append(val)
    return epochs, values


def plot_train_loss(
    ax: plt.Axes,
    epochs: Sequence[int],
    losses: Sequence[float],
    label: Optional[str] = None,
) -> None:
    ax.plot(epochs, losses, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_ylim(0, 5)
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_loss_decomposition_grid(
    history: Sequence[Dict[str, Any]],
    run_label: str,
    output_path: Path,
    splits: Sequence[str] = ("train_eval", "test_eval", "combined_eval"),
    statuses: Sequence[str] = ("observed", "masked", "missing"),
    types: Sequence[str] = ("rating", "ranking_pairwise"),
) -> None:
    """Plot per-split loss decomposition curves for a single run."""
    # Determine which splits are actually present.
    present_splits: List[str] = []
    for split_key in splits:
        for entry in history:
            if split_key in entry:
                present_splits.append(split_key)
                break
    if not present_splits:
        return

    n_splits = len(present_splits)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 4), squeeze=False)

    for idx, split_key in enumerate(present_splits):
        ax = axes[0, idx]
        for type_name in types:
            for status in statuses:
                epochs, values = extract_status_type_metric(
                    history, split_key=split_key, status=status, type_name=type_name, metric_key="xent"
                )
                if not epochs:
                    continue
                curve_label = f"{split_key}.{status}.{type_name}"
                ax.plot(epochs, values, label=curve_label)

        ax.set_title(split_key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy (xent)")
        ax.set_ylim(0, 5)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.suptitle(f"Loss decomposition: {run_label}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def discover_run_dirs(runs_root: Path, explicit_runs: Optional[Iterable[str]]) -> List[Path]:
    """Resolve run directories either from an explicit list or by scanning a root."""
    if explicit_runs:
        dirs: List[Path] = []
        for r in explicit_runs:
            p = Path(r)
            if not p.is_absolute():
                p = runs_root / p
            if (p / "training_history.json").exists():
                dirs.append(p)
        return sorted(dirs)

    if not runs_root.exists():
        return []
    dirs = [
        d
        for d in runs_root.iterdir()
        if d.is_dir() and (d / "training_history.json").exists()
    ]
    return sorted(dirs)


def load_runs(runs_root: Path, explicit_runs: Optional[Iterable[str]]) -> List[RunData]:
    """Load RunData objects from the given root and/or explicit run paths."""
    run_dirs = discover_run_dirs(runs_root, explicit_runs)
    runs: List[RunData] = []
    for rd in run_dirs:
        try:
            history = load_training_history(rd)
        except FileNotFoundError:
            continue
        config = load_train_config(rd)
        label = summarize_run_label(config, rd)
        runs.append(RunData(run_dir=rd, history=history, config=config, label=label))
    return runs


def plot_per_run(runs: Sequence[RunData], output_root: Path) -> None:
    """Generate per-run plots: train loss and loss decomposition."""
    for run in runs:
        epochs, losses = extract_total_loss(run.history)
        if epochs and losses:
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_train_loss(ax, epochs, losses, label=run.label)
            ax.set_title(f"Train loss: {run.label}")
            if run.label:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
            fig.tight_layout()
            out_path = output_root / run.run_dir.name / "train_loss.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)

        # Loss decomposition grid
        out_decomp = output_root / run.run_dir.name / "loss_decomposition.png"
        plot_loss_decomposition_grid(run.history, run.label, out_decomp)


def plot_multi_run_total_loss(runs: Sequence[RunData], output_path: Path) -> None:
    """Plot total train loss vs epoch for multiple runs on the same axes."""
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        epochs, losses = extract_total_loss(run.history)
        if not epochs:
            continue
        plot_train_loss(ax, epochs, losses, label=run.label)
    ax.set_title("Total train loss across runs")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_metric_spec(metric: str) -> Tuple[str, str, str, str]:
    """
    Parse a metric spec of the form 'split/status/type/metric',
    e.g. 'test_eval/missing/rating/xent'.
    """
    parts = metric.split("/")
    if len(parts) != 4:
        raise ValueError(f"Metric spec should be 'split/status/type/metric', got: {metric}")
    return parts[0], parts[1], parts[2], parts[3]


def plot_multi_run_metric(
    runs: Sequence[RunData],
    metric_spec: str,
    output_path: Path,
) -> None:
    """Plot a specific metric across runs, e.g. test_eval/missing/rating/xent."""
    if not runs:
        return

    split_key, status, type_name, metric_key = parse_metric_spec(metric_spec)
    fig, ax = plt.subplots(figsize=(7, 5))

    for run in runs:
        epochs, values = extract_status_type_metric(
            run.history,
            split_key=split_key,
            status=status,
            type_name=type_name,
            metric_key=metric_key,
        )
        if not epochs:
            continue
        ax.plot(epochs, values, label=run.label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{metric_spec} across runs")
    ax.set_ylim(0, 5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_multi_run_combined_or_test_missing_rating_xent(
    runs: Sequence[RunData],
    output_path: Path,
) -> None:
    """
    Plot a unified curve per run where:
    - For transductive runs, we use combined_eval/missing/rating/xent.
    - For non-transductive runs, we fall back to test_eval/missing/rating/xent.
    """
    if not runs:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for run in runs:
        # Try combined_eval first.
        epochs, values = extract_status_type_metric(
            run.history,
            split_key="combined_eval",
            status="missing",
            type_name="rating",
            metric_key="xent",
        )
        # Fall back to test_eval if no combined_eval metrics are present.
        if not epochs:
            epochs, values = extract_status_type_metric(
                run.history,
                split_key="test_eval",
                status="missing",
                type_name="rating",
                metric_key="xent",
            )
        if not epochs:
            continue
        ax.plot(epochs, values, label=run.label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("xent")
    ax.set_title("missing.rating.xent (combined for transductive, test for non-transductive)")
    ax.set_ylim(0, 5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Entity Marformer training history from one or more runs. "
            "Each run directory must contain training_history.json (and optionally train_config.json)."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="OUTPUT/ENTITY_MF",
        help="Root directory containing multiple Entity Marformer runs.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of run directories (relative to runs-root or absolute).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to write plots into. "
            "Defaults to '<runs-root>/plots' for multi-run plots and per-run subdirectories under that."
        ),
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Generate per-run plots (train loss and loss decomposition).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate multi-run comparison plots.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_eval/missing/rating/xent",
        help=(
            "Metric spec for --compare plots, of the form 'split/status/type/metric', "
            "e.g. 'test_eval/missing/rating/xent'."
        ),
    )

    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    runs = load_runs(runs_root, args.runs)
    if not runs:
        print(f"No runs with training_history.json found under {runs_root}")
        return

    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    else:
        output_root = runs_root / "plots"

    if args.per_run:
        plot_per_run(runs, output_root)

    if args.compare:
        # Total train loss comparison.
        total_loss_path = output_root / "multi_run_total_loss.png"
        plot_multi_run_total_loss(runs, total_loss_path)
        print(f"Plotted total train loss to {total_loss_path}")

        # Specific metric comparison (e.g. test_eval/missing/rating/xent).
        metric_safe = args.metric.replace("/", "_")
        metric_path = output_root / f"multi_run_{metric_safe}.png"
        plot_multi_run_metric(runs, args.metric, metric_path)
        print(f"Plotted {args.metric} to {metric_path}")

        # For transductive runs, also plot combined_eval/missing/rating/xent across runs.
        combined_safe = "combined_or_test_eval_missing_rating_xent"
        combined_path = output_root / f"multi_run_{combined_safe}.png"
        plot_multi_run_combined_or_test_missing_rating_xent(runs, combined_path)
        print(f"Plotted {combined_safe} to {combined_path}")

if __name__ == "__main__":
    main()

