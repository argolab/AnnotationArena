"""
``data_bundle.json`` → prediction examples.

Fit pool: all observed ratings in train, val, and test (transductive).
Eval: missing cells with sources = ALL transductive observed cells (train + val + test),
excluding the target cell itself. Sources are NOT restricted to same-item or same-split
neighbors; every observed cell in the transductive pool is a potential source. The split
label only gates which *missing* rows are evaluated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from .cli_defaults import TRANSDUCTIVE_INSTANCES

Cell = Tuple[int, int, int, int]  # (i, j, k, v) 0-based


@dataclass(frozen=True)
class LocalExample:
    target_i: int
    target_j: int
    target_k: int
    y: int
    sources: Tuple[Cell, ...]


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
    all_r = bundle["observed_ratings"] + bundle["missing_ratings"]
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


def transductive_observed_cells(bundle: dict) -> List[Cell]:
    """All observed cells in train, val, and test."""
    return [
        _rating_to_cell(r)
        for r in bundle["observed_ratings"]
        if str(r["instance"]) in TRANSDUCTIVE_INSTANCES
    ]


def transductive_observed_rows(bundle: dict) -> List[dict]:
    return [
        r
        for r in bundle["observed_ratings"]
        if str(r["instance"]) in TRANSDUCTIVE_INSTANCES
    ]


def build_test_examples(bundle: dict) -> List[LocalExample]:
    all_obs = transductive_observed_cells(bundle)

    examples: List[LocalExample] = []
    for r in bundle["missing_ratings"]:
        if str(r["instance"]) != "test":
            continue
        ti_i, ti_j, ti_k, y = _rating_to_cell(r)
        srcs = [c for c in all_obs if (c[0], c[1], c[2]) != (ti_i, ti_j, ti_k)]
        examples.append(
            LocalExample(target_i=ti_i, target_j=ti_j, target_k=ti_k, y=y, sources=tuple(srcs))
        )
    return examples


def build_eval_examples(
    bundle: dict,
    split: Literal["train", "val", "test"],
) -> List[LocalExample]:
    all_obs = transductive_observed_cells(bundle)

    examples: List[LocalExample] = []
    for r in bundle["missing_ratings"]:
        if str(r["instance"]) != split:
            continue
        ti_i, ti_j, ti_k, y = _rating_to_cell(r)
        srcs = [c for c in all_obs if (c[0], c[1], c[2]) != (ti_i, ti_j, ti_k)]
        examples.append(
            LocalExample(target_i=ti_i, target_j=ti_j, target_k=ti_k, y=y, sources=tuple(srcs))
        )
    return examples


def build_train_observed_examples(bundle: dict) -> List[LocalExample]:
    """
    One LocalExample per **observed** train rating: predict that cell from all
    other transductive observed cells (train + val + test).

    Used to supervise structured log-linear when there are no train-**missing** rows
    (e.g. LLM Rubric: humans only on val/test items).
    """
    all_obs = transductive_observed_cells(bundle)

    examples: List[LocalExample] = []
    for r in bundle["observed_ratings"]:
        if str(r["instance"]) != "train":
            continue
        ti_i, ti_j, ti_k, y = _rating_to_cell(r)
        srcs = [c for c in all_obs if (c[0], c[1], c[2]) != (ti_i, ti_j, ti_k)]
        examples.append(
            LocalExample(target_i=ti_i, target_j=ti_j, target_k=ti_k, y=y, sources=tuple(srcs))
        )
    return examples
