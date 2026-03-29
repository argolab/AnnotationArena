#!/usr/bin/env python3
"""
One-off script: learning curves for subtree-SUM task.

Data from ``toy_final.build_random_bounded_tree``: each node draws a scalar
``a_i ~ Uniform(scalar_low, scalar_high)`` (default [-10, 10]); input is ``a_i``;
target at ``u`` is ``sum_{v in subtree(u)} a_v``.

**Sanity / counting mode:** set ``SANITY_ALL_ONES = True`` in this file, or run with
``--sanity-counting``. Then every ``a_i = 1``, targets are **subtree node counts**
(classic counting task) to compare against the random-scalar sum task.

Produces two PNGs (train MSE, test MSE) with 16 curves:
  - num_layers in {1,2,3,4}
  - four architecture presets (see SETTINGS)

Uses ONE fixed train/test dataset for all runs (same RNG seed).
Trains each curve for NUM_STEPS steps.

Run from the ``ranking`` tree:

  cd imputer/ranking
  PYTHONPATH=. python toy_scripts/toy_sum_task_curves_plot.py

Counting sanity (writes to ``OUTPUT/toy_sum_curves_sanity/``):

  PYTHONPATH=. python toy_scripts/toy_sum_task_curves_plot.py --sanity-counting

Train/test MSE plots use a **log-scale y-axis** (same as average curves).
Redraw PNGs from saved JSON without retraining:

  PYTHONPATH=. python toy_scripts/toy_sum_task_curves_plot.py --replot OUTPUT/toy_sum_curves/curves.json

Default outputs: ``OUTPUT/toy_sum_curves/`` (random scalars) or
``OUTPUT/toy_sum_curves_sanity/`` (counting sanity).

Safe to delete after you share plots.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Allow `import toy_final` when launched from any cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from tqdm.auto import tqdm

from imputer.entity_mf.config import EntityMarformerConfig
from imputer.entity_mf.model import EntityMarformer

from toy_final import (
    TreeSample,
    _compute_deviation_reg_loss,
    _compute_mse_on_sample,
    build_random_bounded_tree,
)

# -----------------------------------------------------------------------------
# Experiment controls (match toy_final / toy_average_task_curves_plot structure)
# -----------------------------------------------------------------------------
DATA_SEED = 42
NUM_STEPS = 800
NUM_TRAIN = 100
NUM_TEST = 25
MAX_DEPTH = 4
MAX_DEGREE = 3
MAX_NODES = 32
SHUFFLE_NODES = True
SCALAR_LOW = -10.0
SCALAR_HIGH = 10.0

# If True (or use CLI ``--sanity-counting``): all node scalars = 1.0 => targets are subtree *sizes* (counting).
SANITY_ALL_ONES = False

# Fixed model/training (aligned with toy_average_task_curves_plot.py)
EMBEDDING_DIM = 8
ATTENTION_HEADS = 1
DROPOUT = 0.0
D_FF = 32
NUM_FFN_LAYERS = 1
LR = 5e-3
USE_FEATURE_ONLY_NORM = True
TYPE_EMBEDDING_INIT = "normal"
FREEZE_VARIATION = False

# num_layers -> colormap (4 settings get progressively darker shades within each).
# L=2 vs L=3: use Blues vs Oranges (YlOrBr+Oranges looked too similar).
LAYER_CMAPS = (
    plt.cm.Greens,
    plt.cm.Blues,
    plt.cm.Oranges,
    plt.cm.Purples,
)

SETTINGS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "base",
        {
            "use_rel_value": False,
            "use_addone_attn": False,
            "use_per_head_rel": True,
            "scale_shared_rel": False,
        },
    ),
    (
        "add1",
        {
            "use_rel_value": False,
            "use_addone_attn": True,
            "use_per_head_rel": True,
            "scale_shared_rel": False,
        },
    ),
    (
        "rel",
        {
            "use_rel_value": True,
            "use_addone_attn": False,
            "use_per_head_rel": False,
            "scale_shared_rel": True,
        },
    ),
    (
        "both",
        {
            "use_rel_value": True,
            "use_addone_attn": True,
            "use_per_head_rel": False,
            "scale_shared_rel": True,
        },
    ),
]

NUM_LAYERS_LIST = [1, 2, 3, 4]
DEFAULT_OUT_DIR = Path("OUTPUT/toy_sum_curves")
SANITY_OUT_DIR = Path("OUTPUT/toy_sum_curves_sanity")

# Floor for log-scale plots (MSE should be > 0; avoids log(0))
_LOG_Y_FLOOR = 1e-12


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_fixed_dataset(
    device: torch.device,
    seed: int,
    *,
    sanity_all_ones: bool = False,
) -> Tuple[List[TreeSample], List[TreeSample]]:
    """Same protocol as toy_final.run_entity_mf_random_trees: one RNG for all graphs."""
    rng = random.Random(seed)
    train_samples: List[TreeSample] = []
    for _ in range(NUM_TRAIN):
        train_samples.append(
            build_random_bounded_tree(
                device=device,
                rng=rng,
                max_depth=MAX_DEPTH,
                max_degree=MAX_DEGREE,
                max_nodes=MAX_NODES,
                shuffle_nodes=SHUFFLE_NODES,
                scalar_low=SCALAR_LOW,
                scalar_high=SCALAR_HIGH,
                sanity_all_ones=sanity_all_ones,
            )
        )
    test_samples: List[TreeSample] = []
    for _ in range(NUM_TEST):
        test_samples.append(
            build_random_bounded_tree(
                device=device,
                rng=rng,
                max_depth=MAX_DEPTH,
                max_degree=MAX_DEGREE,
                max_nodes=MAX_NODES,
                shuffle_nodes=SHUFFLE_NODES,
                scalar_low=SCALAR_LOW,
                scalar_high=SCALAR_HIGH,
                sanity_all_ones=sanity_all_ones,
            )
        )
    return train_samples, test_samples


def run_one_curve(
    device: torch.device,
    train_samples: List[TreeSample],
    test_samples: List[TreeSample],
    num_layers: int,
    preset: Dict[str, Any],
    *,
    progress_desc: str | None = None,
) -> Tuple[List[float], List[float]]:
    """Returns (train_mse per step, test_mse per step), length NUM_STEPS."""
    ref_graph = train_samples[0].graph
    types = ref_graph.types

    cfg = EntityMarformerConfig(
        embedding_dim=EMBEDDING_DIM,
        num_layers=num_layers,
        attention_heads=ATTENTION_HEADS,
        dropout=DROPOUT,
        d_ff=D_FF,
        num_ffn_layers=NUM_FFN_LAYERS,
        use_per_head_rel=bool(preset["use_per_head_rel"]),
        use_rel_value=bool(preset["use_rel_value"]),
        use_addone_attn=bool(preset["use_addone_attn"]),
        use_feature_only_norm=USE_FEATURE_ONLY_NORM,
        scale_shared_rel=bool(preset["scale_shared_rel"]),
        type_embedding_init=TYPE_EMBEDDING_INIT,
    )

    model = EntityMarformer(
        config=cfg,
        types=types,
        num_relationships=ref_graph.num_relationships,
    ).to(device)

    if FREEZE_VARIATION:
        for type_name, t in types.items():
            if not t.variation.enabled or t.variation.num_entities <= 0:
                continue
            table = model.deviation_tables.get(type_name, None)
            if table is None:
                continue
            with torch.no_grad():
                table.normal_(mean=0.0, std=0.1)
            table.requires_grad_(False)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    train_curve: List[float] = []
    test_curve: List[float] = []

    steps = range(NUM_STEPS)
    if progress_desc is not None:
        steps = tqdm(steps, total=NUM_STEPS, desc=progress_desc, leave=False)

    for _step in steps:
        model.train()
        opt.zero_grad(set_to_none=True)

        mse_sum = torch.zeros((), device=device)
        for sample in train_samples:
            mse_sum = mse_sum + _compute_mse_on_sample(model, sample, device=device)
        train_mse = mse_sum / float(len(train_samples))

        reg_loss = _compute_deviation_reg_loss(model, types, device=device)
        loss = train_mse + reg_loss
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            test_sum = torch.zeros((), device=device)
            for sample in test_samples:
                test_sum = test_sum + _compute_mse_on_sample(model, sample, device=device)
            test_mse = test_sum / float(len(test_samples))

        train_curve.append(float(train_mse.detach().cpu().item()))
        test_curve.append(float(test_mse.detach().cpu().item()))

    return train_curve, test_curve


def _color_for_curve(layer_idx: int, setting_idx: int) -> Tuple[float, float, float, float]:
    cmap = LAYER_CMAPS[layer_idx]
    t = 0.35 + 0.55 * (setting_idx / max(1, len(SETTINGS) - 1))
    rgba = cmap(t)
    return rgba


def render_plots_from_results(
    results: Dict[str, Any],
    out_dir: Path,
    *,
    log_y: bool = True,
) -> None:
    """
    Draw train/test MSE figures from a ``curves.json``-compatible dict.
    When ``log_y`` is True, y-axis is log-scaled (values clipped to >= _LOG_Y_FLOOR).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    num_steps = int(results.get("num_steps", NUM_STEPS))
    x = np.arange(1, num_steps + 1)

    if results.get("sanity_all_ones"):
        title_mid = "subtree node count (sanity: all inputs=1)"
    else:
        sr = results.get("scalar_range", [SCALAR_LOW, SCALAR_HIGH])
        lo, hi = float(sr[0]), float(sr[1])
        title_mid = f"subtree sum, U[{lo},{hi}] scalars"

    fig_train, ax_tr = plt.subplots(figsize=(10, 6))
    fig_test, ax_te = plt.subplots(figsize=(10, 6))

    for li, num_layers in enumerate(NUM_LAYERS_LIST):
        for si, (sname, _preset) in enumerate(SETTINGS):
            key = f"L{num_layers}_{sname}"
            if key not in results.get("curves", {}):
                continue
            c = results["curves"][key]
            tr = np.asarray(c["train_mse"], dtype=np.float64)
            te = np.asarray(c["test_mse"], dtype=np.float64)
            if log_y:
                tr = np.maximum(tr, _LOG_Y_FLOOR)
                te = np.maximum(te, _LOG_Y_FLOOR)
            color = _color_for_curve(li, si)
            label = f"L={num_layers} · {sname}"
            ax_tr.plot(x, tr, color=color, linewidth=1.8, label=label)
            ax_te.plot(x, te, color=color, linewidth=1.8, label=label)

    log_note = " — log y" if log_y else ""
    y_lab = "train MSE (log scale)" if log_y else "train MSE"
    y_te_lab = "test MSE (log scale)" if log_y else "test MSE"

    for ax, title_prefix, ylabel, fname in (
        (ax_tr, f"Train MSE ({title_mid}){log_note}", y_lab, "sum_task_train_mse.png"),
        (ax_te, f"Test MSE ({title_mid}){log_note}", y_te_lab, "sum_task_test_mse.png"),
    ):
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title_prefix)
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(
            fontsize=7,
            ncol=2,
            loc="upper right",
            framealpha=0.92,
        )
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def render_layer_split_test_plots_from_results(
    results: Dict[str, Any],
    out_dir: Path,
    *,
    log_y: bool = True,
) -> None:
    """
    Create 4 side-by-side (train/test) plots, one per num_layers in NUM_LAYERS_LIST.

    Each plot shows only the 4 architecture presets (SETTINGS) for that layer
    setting, and uses the same color mapping as the master plot (via
    _color_for_curve(layer_idx, setting_idx)).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    num_steps = int(results.get("num_steps", NUM_STEPS))
    x = np.arange(1, num_steps + 1)

    if results.get("sanity_all_ones"):
        title_mid = "subtree node count (sanity: all inputs=1)"
    else:
        sr = results.get("scalar_range", [SCALAR_LOW, SCALAR_HIGH])
        lo, hi = float(sr[0]), float(sr[1])
        title_mid = f"subtree sum, U[{lo},{hi}] scalars"

    y_te_lab = "test MSE (log scale)" if log_y else "test MSE"
    log_note = " — log y" if log_y else ""

    for li, num_layers in enumerate(NUM_LAYERS_LIST):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_tr, ax_te = axes
        for si, (sname, _preset) in enumerate(SETTINGS):
            key = f"L{num_layers}_{sname}"
            if key not in results.get("curves", {}):
                continue
            c = results["curves"][key]
            tr = np.asarray(c["train_mse"], dtype=np.float64)
            te = np.asarray(c["test_mse"], dtype=np.float64)
            if log_y:
                tr = np.maximum(tr, _LOG_Y_FLOOR)
                te = np.maximum(te, _LOG_Y_FLOOR)
            color = _color_for_curve(li, si)
            label = f"{sname}"
            ax_tr.plot(x, tr, color=color, linewidth=2.0, label=label)
            ax_te.plot(x, te, color=color, linewidth=2.0, label=label)

        ax_tr.set_xlabel("step")
        ax_tr.set_ylabel("train MSE (log scale)" if log_y else "train MSE")
        ax_tr.set_title(f"Train MSE (L={num_layers})")
        if log_y:
            ax_tr.set_yscale("log")
        ax_tr.grid(True, which="both", alpha=0.3)

        ax_te.set_xlabel("step")
        ax_te.set_ylabel(y_te_lab)
        ax_te.set_title(f"Test MSE by layer (L={num_layers}) ({title_mid}){log_note}")
        if log_y:
            ax_te.set_yscale("log")
        ax_te.grid(True, which="both", alpha=0.3)

        # Shared legend (use handles from train axis).
        ax_te.legend(fontsize=8, loc="upper right", framealpha=0.92)
        fig.tight_layout()
        fig.savefig(out_dir / f"sum_task_test_mse_L{num_layers}.png", dpi=160)
        plt.close(fig)


def main(
    *,
    sanity_all_ones: bool | None = None,
    out_dir: Path | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_seed(DATA_SEED)

    san = SANITY_ALL_ONES if sanity_all_ones is None else sanity_all_ones
    out = out_dir if out_dir is not None else (SANITY_OUT_DIR if san else DEFAULT_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    train_samples, test_samples = build_fixed_dataset(
        device=device, seed=DATA_SEED, sanity_all_ones=san
    )

    results: Dict[str, Any] = {
        "task": "subtree_count_sanity" if san else "subtree_sum_scalars",
        "sanity_all_ones": san,
        "scalar_range": [SCALAR_LOW, SCALAR_HIGH],
        "num_steps": NUM_STEPS,
        "data_seed": DATA_SEED,
        "num_train": NUM_TRAIN,
        "num_test": NUM_TEST,
        "curves": {},
    }

    for num_layers in NUM_LAYERS_LIST:
        for sname, preset in SETTINGS:
            key = f"L{num_layers}_{sname}"
            tr, te = run_one_curve(
                device=device,
                train_samples=train_samples,
                test_samples=test_samples,
                num_layers=num_layers,
                preset=preset,
                progress_desc=key,
            )
            results["curves"][key] = {
                "num_layers": num_layers,
                "preset": sname,
                **preset,
                "train_mse": tr,
                "test_mse": te,
            }

    (out / "curves.json").write_text(json.dumps(results, indent=2))
    render_plots_from_results(results, out, log_y=True)
    render_layer_split_test_plots_from_results(results, out, log_y=True)
    print(f"Wrote log-scale plots and {out / 'curves.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum-task learning curves (log y-axis plots).")
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        metavar="PATH",
        help="Only redraw PNGs from an existing curves.json (no training). Example: OUTPUT/toy_sum_curves/curves.json",
    )
    parser.add_argument(
        "--sanity-counting",
        action="store_true",
        help="All node scalars=1.0 (subtree targets = node counts). Output dir: OUTPUT/toy_sum_curves_sanity/",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Override output directory for curves.json and PNGs (default: toy_sum_curves or toy_sum_curves_sanity).",
    )
    args = parser.parse_args()
    if args.replot:
        json_path = Path(args.replot)
        results = json.loads(json_path.read_text())
        render_plots_from_results(results, json_path.parent, log_y=True)
        render_layer_split_test_plots_from_results(results, json_path.parent, log_y=True)
        print(f"Wrote log-scale plots to {json_path.parent}")
    else:
        san = SANITY_ALL_ONES or args.sanity_counting
        od = Path(args.out_dir) if args.out_dir else None
        main(sanity_all_ones=san, out_dir=od)
