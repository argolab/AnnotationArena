"""
Convert data_bundle.json-style dicts into local prediction examples.

Bundle convention (matches imputer/ranking elsewhere):
  - attribute, annotator, item, value are 1-indexed integers in JSON.
  - We convert to 0-based (i,j,k,v) inside LocalExample.

A *plate* is all rating cells that share the same (item, instance) key.
Training uses leave-one-out on each plate from selected instances (default: train only).
Test uses each test missing rating as target with sources = observed cells on the same
(item, test) plate — no leakage from held-out values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

PlateKey = Tuple[int, str]  # (item_1idx, instance)
Cell = Tuple[int, int, int, int]  # (i,j,k,v) all 0-based


@dataclass(frozen=True)
class LocalExample:
    """Predict one target cell from a list of source cells on the same item plate."""

    target_i: int
    target_j: int
    target_k: int
    y: int  # 0-based class
    sources: Tuple[Cell, ...]  # each (i,j,k,v) 0-based, excludes the target cell


def _rating_to_cell(r: dict) -> Cell:
    return (
        int(r["attribute"]) - 1,
        int(r["annotator"]) - 1,
        int(r["item"]) - 1,
        int(r["value"]) - 1,
    )


def load_bundle_dict(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def bundle_dims(bundle: dict, bundle_path: Optional[Path] = None) -> Tuple[int, int, int]:
    """
    Return (I, J, num_classes) as cardinalities inferred from the bundle + configs.json.

    I = number of distinct attribute ids, J = distinct annotator ids,
    num_classes = C from configs.datagen.C if present else max value in ratings.
    """
    all_r = bundle["observed_ratings"] + bundle["missing_ratings"]
    # 1-based ids → array sizes must cover max index
    I = max(int(r["attribute"]) for r in all_r)
    J = max(int(r["annotator"]) for r in all_r)
    C: Optional[int] = None
    if bundle_path is not None:
        cfgp = Path(bundle_path).parent / "configs.json"
        if cfgp.exists():
            with open(cfgp) as f:
                cfg = json.load(f)
            C = cfg.get("datagen", cfg).get("C")
    if C is None:
        C = max(int(r["value"]) for r in all_r)
    return I, J, int(C)


def _group_plates(
    bundle: dict,
    instances: Set[str],
) -> Dict[PlateKey, List[Cell]]:
    """Merge observed + missing ratings for each (item, instance)."""
    plates: Dict[PlateKey, List[Cell]] = {}
    for r in bundle["observed_ratings"]:
        inst = str(r["instance"])
        if inst not in instances:
            continue
        key = (int(r["item"]), inst)
        plates.setdefault(key, []).append(_rating_to_cell(r))
    for r in bundle["missing_ratings"]:
        inst = str(r["instance"])
        if inst not in instances:
            continue
        key = (int(r["item"]), inst)
        plates.setdefault(key, []).append(_rating_to_cell(r))
    return plates


def build_training_examples(
    bundle: dict,
    instances: Set[str] | None = None,
) -> List[LocalExample]:
    """
    Leave-one-out examples: on each plate, each cell is a target once; sources are the others.

    Default instances == {"train"} so we do not peek at val/test plates during fitting.
    Pass {"train", "val"} if you want more supervision (still no test).
    """
    if instances is None:
        instances = {"train"}
    examples: List[LocalExample] = []
    for _key, cells in _group_plates(bundle, instances).items():
        if len(cells) < 2:
            continue
        n = len(cells)
        for ti in range(n):
            ti_i, ti_j, ti_k, y = cells[ti]
            srcs: List[Cell] = []
            for sj in range(n):
                if sj == ti:
                    continue
                srcs.append(cells[sj])
            examples.append(
                LocalExample(
                    target_i=ti_i,
                    target_j=ti_j,
                    target_k=ti_k,
                    y=y,
                    sources=tuple(srcs),
                )
            )
    return examples


def build_test_examples(bundle: dict) -> List[LocalExample]:
    """
    One example per test missing cell: sources are test-observed cells on the same item.

    Ground-truth y is the held-out value from missing_ratings (still 0-based in LocalExample).
    """
    obs_by_item: Dict[int, List[Cell]] = {}
    for r in bundle["observed_ratings"]:
        if str(r["instance"]) != "test":
            continue
        item = int(r["item"])
        obs_by_item.setdefault(item, []).append(_rating_to_cell(r))

    examples: List[LocalExample] = []
    for r in bundle["missing_ratings"]:
        if str(r["instance"]) != "test":
            continue
        item = int(r["item"])
        ti_i, ti_j, ti_k, y = _rating_to_cell(r)
        srcs = [c for c in obs_by_item.get(item, []) if (c[0], c[1], c[2]) != (ti_i, ti_j, ti_k)]
        examples.append(
            LocalExample(
                target_i=ti_i,
                target_j=ti_j,
                target_k=ti_k,
                y=y,
                sources=tuple(srcs),
            )
        )
    return examples


def build_eval_examples(
    bundle: dict,
    split: Literal["train", "val", "test"],
) -> List[LocalExample]:
    """
    Like test builder but for any split: target = missing cells, sources = observed on same item.

    Useful for train/val diagnostics without touching test missing definitions.
    """
    obs_by_item: Dict[int, List[Cell]] = {}
    for r in bundle["observed_ratings"]:
        if str(r["instance"]) != split:
            continue
        item = int(r["item"])
        obs_by_item.setdefault(item, []).append(_rating_to_cell(r))

    examples: List[LocalExample] = []
    for r in bundle["missing_ratings"]:
        if str(r["instance"]) != split:
            continue
        item = int(r["item"])
        ti_i, ti_j, ti_k, y = _rating_to_cell(r)
        srcs = [c for c in obs_by_item.get(item, []) if (c[0], c[1], c[2]) != (ti_i, ti_j, ti_k)]
        examples.append(
            LocalExample(
                target_i=ti_i,
                target_j=ti_j,
                target_k=ti_k,
                y=y,
                sources=tuple(srcs),
            )
        )
    return examples


def ratings_for_ijk_fit(
    bundle: dict,
    transductive: bool,
    instances: Optional[Set[str]] = None,
) -> List[dict]:
    """
    Flat list of raw rating dicts used by the classic IJK Naive Bayes pool.

    If transductive, uses observed rows whose instance is train, val, or test.
    If not, uses train and val only (excludes test-observed from the count pool).
    """
    if instances is not None:
        inst_filter = instances
    else:
        inst_filter = {"train", "val", "test"} if transductive else {"train", "val"}
    return [r for r in bundle["observed_ratings"] if str(r["instance"]) in inst_filter]
