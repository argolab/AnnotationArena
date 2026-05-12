#!/usr/bin/env python3
"""
Plot DOMAIN3-FINAL generalization figures with the existing Domain 3 talk style.

Outputs:
  - PLOTS/DOMAIN3-FINAL/ITEM_GENERALIZATION/*
  - PLOTS/DOMAIN3-FINAL/ANNOT_GENERALIZATION/*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.utils import plot_domain3_generalization as base


def _read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _runtime_from_launcher(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def _marformer_runtime_seconds(root: Path, sizes: list[int], stem: str) -> dict[int, float]:
    runtime_map: dict[int, float] = {}
    for size in sizes:
        run_dir = root / f"{stem}{size}_MARFORMER"
        value = _runtime_from_launcher(run_dir / "launcher_metadata.json", "elapsed_seconds")
        if value is not None:
            runtime_map[size] = value
    return runtime_map


def _stan_runtime_seconds(root: Path, sizes: list[int], stem: str, suffix: str) -> dict[int, float]:
    runtime_map: dict[int, float] = {}
    for size in sizes:
        run_dir = root / f"{stem}{size}_{suffix}"
        value = _runtime_from_launcher(run_dir / "launcher_metadata.json", "fit_elapsed_seconds")
        if value is not None:
            runtime_map[size] = value
    return runtime_map


def _misspec_runtime_seconds(root: Path, sizes: list[int], stem: str, suffix: str) -> dict[int, float]:
    runtime_map: dict[int, float] = {}
    for size in sizes:
        run_dir = root / f"{stem}{size}_{suffix}"
        value = _runtime_from_launcher(run_dir / "launcher_metadata.json", "fit_elapsed_seconds")
        if value is not None:
            runtime_map[size] = value
    return runtime_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DOMAIN3-FINAL generalization figures.")
    parser.add_argument(
        "--use-summary",
        action="store_true",
        help="Reuse metric/runtime values from saved summary JSON when available instead of recomputing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.MISSPEC_ROOT = ROOT / "RESULTS/STAN/DOMAIN3-FINAL/MISSPEC"
    base.MISSPEC_MODELS = [
        ("Stan Both", "BOTH"),
        ("Stan Bin", "BIN"),
        ("Stan Proj", "PROJ"),
    ]

    item_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    annot_sizes = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    item_marformer_root = ROOT / "RESULTS/MARFORMER/DOMAIN3-FINAL/ITEM"
    annot_marformer_root = ROOT / "RESULTS/MARFORMER/DOMAIN3-FINAL/ANNOT"
    item_marformer_nt_root = ROOT / "RESULTS/MARFORMER-NT/DOMAIN3-FINAL/ITEM"
    annot_marformer_nt_root = ROOT / "RESULTS/MARFORMER-NT/DOMAIN3-FINAL/ANNOT"
    item_oracle_root = ROOT / "RESULTS/STAN/DOMAIN3-FINAL/ORACLE/ITEM"
    annot_oracle_root = ROOT / "RESULTS/STAN/DOMAIN3-FINAL/ORACLE/ANNOT"
    item_misspec_root = ROOT / "RESULTS/STAN/DOMAIN3-FINAL/MISSPEC/ITEM"
    annot_misspec_root = ROOT / "RESULTS/STAN/DOMAIN3-FINAL/MISSPEC/ANNOT"

    base.ITEM_SPEC = base.DatasetSpec(
        slug="item",
        title_prefix="Item Generalization",
        sizes=item_sizes,
        x_label="Training Items",
        data_root=ROOT / "DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive",
        marformer_root=item_marformer_root,
        stan_root=item_oracle_root,
        marformer_run_glob="DOMAIN3-FINAL_Item_T_{size}_MARFORMER*",
        stan_eval_name="DOMAIN3-FINAL_Item_T_{size}_SHARED_THRESH_ORACLE_eval",
        out_dir=ROOT / "PLOTS/DOMAIN3-FINAL/ITEM_GENERALIZATION",
        marformer_total_runtime_seconds=_marformer_runtime_seconds(
            item_marformer_root, item_sizes, "DOMAIN3-FINAL_Item_T_"
        ),
        stan_runtime_seconds=_stan_runtime_seconds(
            item_oracle_root, item_sizes, "DOMAIN3-FINAL_Item_T_", "SHARED_THRESH_ORACLE"
        ),
        figure_size=(17.4, 7.2),
        marformer_nt_root=item_marformer_nt_root,
        marformer_nt_run_glob="DOMAIN3-FINAL_Item_NT_{size}_MARFORMER*",
        nt_offset=100,
        misspec_runtime_seconds={
            "Stan Both": _misspec_runtime_seconds(item_misspec_root, item_sizes, "DOMAIN3-FINAL_Item_T_", "BOTH"),
            "Stan Bin": _misspec_runtime_seconds(item_misspec_root, item_sizes, "DOMAIN3-FINAL_Item_T_", "BIN"),
            "Stan Proj": _misspec_runtime_seconds(item_misspec_root, item_sizes, "DOMAIN3-FINAL_Item_T_", "PROJ"),
        },
    )

    base.ANNOT_SPEC = base.DatasetSpec(
        slug="annot",
        title_prefix="Annotator Generalization",
        sizes=annot_sizes,
        x_label="Training Annotators",
        data_root=ROOT / "DATA/STAN/DOMAIN3-FINAL/AnnotSplits/Transductive",
        marformer_root=annot_marformer_root,
        stan_root=annot_oracle_root,
        marformer_run_glob="DOMAIN3-FINAL_Annot_T_{size}_MARFORMER*",
        stan_eval_name="DOMAIN3-FINAL_Annot_T_{size}_SHARED_THRESH_ORACLE_eval",
        out_dir=ROOT / "PLOTS/DOMAIN3-FINAL/ANNOT_GENERALIZATION",
        marformer_total_runtime_seconds=_marformer_runtime_seconds(
            annot_marformer_root, annot_sizes, "DOMAIN3-FINAL_Annot_T_"
        ),
        stan_runtime_seconds=_stan_runtime_seconds(
            annot_oracle_root, annot_sizes, "DOMAIN3-FINAL_Annot_T_", "SHARED_THRESH_ORACLE"
        ),
        figure_size=(17.0, 7.2),
        marformer_nt_root=annot_marformer_nt_root,
        marformer_nt_run_glob="DOMAIN3-FINAL_Annot_NT_{size}_MARFORMER*",
        nt_offset=10,
        misspec_runtime_seconds={
            "Stan Both": _misspec_runtime_seconds(annot_misspec_root, annot_sizes, "DOMAIN3-FINAL_Annot_T_", "BOTH"),
            "Stan Bin": _misspec_runtime_seconds(annot_misspec_root, annot_sizes, "DOMAIN3-FINAL_Annot_T_", "BIN"),
            "Stan Proj": _misspec_runtime_seconds(annot_misspec_root, annot_sizes, "DOMAIN3-FINAL_Annot_T_", "PROJ"),
        },
    )

    base.plot_item(use_summary=args.use_summary)
    base.plot_annot(use_summary=args.use_summary)


if __name__ == "__main__":
    main()
