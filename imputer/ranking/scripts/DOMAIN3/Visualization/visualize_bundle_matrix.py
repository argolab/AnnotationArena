#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np


STATUS_TO_CODE = {
    "unused": 0,
    "train_observed": 1,
    "train_missing": 2,
    "test_observed": 3,
    "test_missing": 4,
    "val_observed": 5,
    "val_missing": 6,
}

CODE_TO_COLOR = [
    "#c7c7c7",  # unused
    "#0b6e2f",  # train observed
    "#9bd49b",  # train missing
    "#2b6cff",  # test observed
    "#d62828",  # test missing
    "#d9a441",  # val observed
    "#f0d7a1",  # val missing
]

LEGEND_ITEMS = [
    ("unused", "Unused / not present"),
    ("train_observed", "Train observed"),
    ("train_missing", "Train missing"),
    ("test_observed", "Test observed"),
    ("test_missing", "Test missing"),
    ("val_observed", "Val observed"),
    ("val_missing", "Val missing"),
]


def load_bundle(bundle_path: Path) -> dict:
    with open(bundle_path) as f:
        return json.load(f)


def get_sizes(bundle: dict) -> tuple[int, int, int]:
    stats = bundle["stats"]
    i_dim = int(stats["I"])
    j_dim = int(stats["J"])
    if "total_items" in stats:
        k_dim = int(stats["total_items"])
    elif "K" in stats:
        k_dim = int(stats["K"])
    else:
        raise ValueError("Could not determine number of items from bundle stats.")
    return i_dim, j_dim, k_dim


def flat_index(attr: int, annotator: int, item: int, j_dim: int, k_dim: int) -> int:
    return ((attr - 1) * j_dim + (annotator - 1)) * k_dim + (item - 1)


def validate_test_missing_indices(bundle: dict, j_dim: int, k_dim: int) -> dict:
    missing = bundle["missing_ratings"]
    field_indices = set(int(x) for x in bundle.get("missing_ratings_indexes_in_test_instance", []))
    row_indices = {idx for idx, row in enumerate(missing) if row["instance"] == "test"}
    flat_indices = {
        flat_index(int(row["attribute"]), int(row["annotator"]), int(row["item"]), j_dim, k_dim)
        for row in missing
        if row["instance"] == "test"
    }
    return {
        "field_count": len(field_indices),
        "row_test_count": len(row_indices),
        "flat_test_count": len(flat_indices),
        "field_equals_row_indices": field_indices == row_indices,
        "field_equals_flat_indices": field_indices == flat_indices,
        "row_equals_flat_indices": row_indices == flat_indices,
        "field_not_in_flat": sorted(field_indices - flat_indices)[:20],
        "flat_not_in_field": sorted(flat_indices - field_indices)[:20],
        "field_min": min(field_indices) if field_indices else None,
        "field_max": max(field_indices) if field_indices else None,
        "flat_min": min(flat_indices) if flat_indices else None,
        "flat_max": max(flat_indices) if flat_indices else None,
    }


def classify(row: dict, observed: bool) -> str:
    prefix = row["instance"]
    suffix = "observed" if observed else "missing"
    key = f"{prefix}_{suffix}"
    if key not in STATUS_TO_CODE:
        raise ValueError(f"Unsupported instance/status combination: {key}")
    return key


def build_attribute_matrices(bundle: dict, i_dim: int, j_dim: int, k_dim: int) -> list[np.ndarray]:
    mats = [np.full((j_dim, k_dim), STATUS_TO_CODE["unused"], dtype=np.int8) for _ in range(i_dim)]
    seen: dict[tuple[int, int, int], str] = {}

    def place(rows: list[dict], observed: bool) -> None:
        for row in rows:
            attr = int(row["attribute"])
            annot = int(row["annotator"])
            item = int(row["item"])
            key = (attr, annot, item)
            label = classify(row, observed)
            if key in seen:
                raise ValueError(f"Duplicate rating cell encountered for {key}: {seen[key]} and {label}")
            seen[key] = label
            mats[attr - 1][annot - 1, item - 1] = STATUS_TO_CODE[label]

    place(bundle["observed_ratings"], observed=True)
    place(bundle["missing_ratings"], observed=False)
    return mats


def item_ticks(k_dim: int) -> list[int]:
    if k_dim <= 20:
        step = 1
    elif k_dim <= 60:
        step = 5
    elif k_dim <= 120:
        step = 10
    else:
        step = 25
    ticks = list(range(1, k_dim + 1, step))
    if ticks[-1] != k_dim:
        ticks.append(k_dim)
    return ticks


def plot_single_attribute(matrix: np.ndarray, attr_idx: int, output_path: Path) -> None:
    cmap = ListedColormap(CODE_TO_COLOR)
    norm = BoundaryNorm(np.arange(len(CODE_TO_COLOR) + 1) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=220)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_title(f"Attribute {attr_idx}", pad=12)
    ax.set_xlabel("Item")
    ax.set_ylabel("Annotator")

    j_dim, k_dim = matrix.shape
    xticks = item_ticks(k_dim)
    ax.set_xticks([x - 1 for x in xticks])
    ax.set_xticklabels([str(x) for x in xticks], fontsize=8)
    ax.set_yticks(range(j_dim))
    ax.set_yticklabels([str(j) for j in range(1, j_dim + 1)], fontsize=8)

    handles = [Patch(facecolor=CODE_TO_COLOR[STATUS_TO_CODE[key]], edgecolor="none", label=label) for key, label in LEGEND_ITEMS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_overview(matrices: list[np.ndarray], output_path: Path) -> None:
    cmap = ListedColormap(CODE_TO_COLOR)
    norm = BoundaryNorm(np.arange(len(CODE_TO_COLOR) + 1) - 0.5, cmap.N)
    i_dim = len(matrices)
    ncols = 3
    nrows = int(np.ceil(i_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.8 * nrows), dpi=220)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, matrix in enumerate(matrices):
        ax = axes[idx // ncols, idx % ncols]
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", origin="upper")
        ax.set_title(f"Attribute {idx + 1}", pad=8)
        ax.set_xlabel("Item")
        ax.set_ylabel("Annotator")
        j_dim, k_dim = matrix.shape
        xticks = item_ticks(k_dim)
        ax.set_xticks([x - 1 for x in xticks])
        ax.set_xticklabels([str(x) for x in xticks], fontsize=7)
        ax.set_yticks(range(j_dim))
        ax.set_yticklabels([str(j) for j in range(1, j_dim + 1)], fontsize=7)

    for idx in range(i_dim, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    handles = [Patch(facecolor=CODE_TO_COLOR[STATUS_TO_CODE[key]], edgecolor="none", label=label) for key, label in LEGEND_ITEMS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_merged_table(matrices: list[np.ndarray], output_path: Path) -> None:
    cmap = ListedColormap(CODE_TO_COLOR)
    norm = BoundaryNorm(np.arange(len(CODE_TO_COLOR) + 1) - 0.5, cmap.N)

    merged = np.vstack(matrices)
    i_dim = len(matrices)
    j_dim, k_dim = matrices[0].shape

    fig_h = max(7, 0.5 * i_dim * j_dim + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_h), dpi=240)
    ax.imshow(merged, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_title("All Attributes Merged", pad=12)
    ax.set_xlabel("Item")
    ax.set_ylabel("Attribute / Annotator")

    xticks = item_ticks(k_dim)
    ax.set_xticks([x - 1 for x in xticks])
    ax.set_xticklabels([str(x) for x in xticks], fontsize=8)

    row_labels = []
    row_positions = []
    for attr_idx in range(i_dim):
        start = attr_idx * j_dim
        center = start + (j_dim - 1) / 2.0
        row_positions.append(center)
        row_labels.append(f"A{attr_idx + 1}")
        if attr_idx > 0:
            ax.axhline(start - 0.5, color="white", linewidth=1.6, alpha=0.95)

    ax.set_yticks(row_positions)
    ax.set_yticklabels(row_labels, fontsize=9)

    # Put annotator ids on the left margin for the first block only as a scale reference.
    for annot_idx in range(j_dim):
        ax.text(
            -3.0,
            annot_idx,
            f"J{annot_idx + 1}",
            ha="right",
            va="center",
            fontsize=8,
            color="#333333",
            clip_on=False,
        )

    handles = [Patch(facecolor=CODE_TO_COLOR[STATUS_TO_CODE[key]], edgecolor="none", label=label) for key, label in LEGEND_ITEMS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize bundle matrices per attribute.")
    parser.add_argument("--bundle", required=True, help="Path to data_bundle.json")
    parser.add_argument("--output-dir", default=None, help="Directory for output PNGs and summary JSON")
    parser.add_argument("--strict-index-check", action="store_true", help="Fail if the test-missing index field does not match dense flat (i,j,k) indices.")
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    bundle = load_bundle(bundle_path)
    i_dim, j_dim, k_dim = get_sizes(bundle)
    output_dir = Path(args.output_dir) if args.output_dir else bundle_path.parent / "matrix_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_test_missing_indices(bundle, j_dim, k_dim)
    summary = {
        "bundle": str(bundle_path),
        "sizes": {"I": i_dim, "J": j_dim, "K": k_dim},
        "validation": validation,
    }

    with open(output_dir / "index_validation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    if args.strict_index_check and not validation["field_equals_flat_indices"]:
        raise SystemExit("Test-missing index field does not match dense flat (i,j,k) indices.")

    matrices = build_attribute_matrices(bundle, i_dim, j_dim, k_dim)
    plot_overview(matrices, output_dir / "all_attributes.png")
    plot_merged_table(matrices, output_dir / "all_attributes_merged.png")
    for attr_idx, matrix in enumerate(matrices, start=1):
        plot_single_attribute(matrix, attr_idx, output_dir / f"attribute_{attr_idx:02d}.png")

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
