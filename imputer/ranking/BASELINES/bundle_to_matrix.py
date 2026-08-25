"""
bundle_to_matrix.py
-------------------
Loads a data_bundle.json and builds item × (annotator * attribute) matrices
for matrix-completion baselines (ReMasker, MIWAE).

Column layout:  col = (annotator_0idx) * I + (attribute_0idx)

Values are 1-indexed integers (matching the bundle `value` field).
Missing entries are NaN in X; M is 1 where observed, 0 where missing.

"Context" annotators at test time (LLM for LLMRubric, turkers for SummEval)
are read directly from the bundle: whichever annotators appear in
observed_ratings[instance='test'] are context; those in
missing_ratings[instance='test'] are the targets we predict.

Usage
-----
    from bundle_to_matrix import load_bundle
    data = load_bundle("path/to/data_bundle.json")

    data.X_train, data.M_train   # (K_train, D) — training items
    data.X_test,  data.M_test    # (K_test,  D) — test items, context cols filled
    data.test_targets            # list of (row, col, value_0idx) for evaluation
    data.context_annotators      # set of 1-indexed ann IDs observed at test time
    data.target_annotators       # set of 1-indexed ann IDs to predict at test time
    data.context_cols_mask       # (D,) bool — True for LLM/turker columns
    data.target_cols_mask        # (D,) bool — True for human/expert columns
    data.meta                    # dict: K_train, K_test, J, I, C, D
    data.col_map                 # {(ann_1idx, attr_1idx): col_idx}
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class BundleMatrix:
    X_train: np.ndarray        # (K_train, D) float32, NaN = missing
    M_train: np.ndarray        # (K_train, D) float32, 1 = observed
    X_test:  np.ndarray        # (K_test,  D) float32, NaN = missing (target cols)
    M_test:  np.ndarray        # (K_test,  D) float32, 1 = observed (context cols)
    X_fit:   np.ndarray        # matrix actually used for baseline fitting
    M_fit:   np.ndarray        # observed-mask for X_fit

    col_map: Dict[Tuple[int, int], int]  # (ann_1idx, attr_1idx) → col_idx
    context_annotators: Set[int]         # 1-indexed ann IDs observed at test time
    target_annotators:  Set[int]         # 1-indexed ann IDs missing at test time
    context_cols_mask: np.ndarray        # (D,) bool — True for LLM/turker columns
    target_cols_mask:  np.ndarray        # (D,) bool — True for human/expert columns

    train_items: List[int]
    test_items:  List[int]
    fit_items:   List[int]

    meta: dict                           # K_train, K_test, J, I, C, D

    test_targets: List[Tuple[int, int, int]]  # (row_idx, col_idx, value_0idx)
    training_mode: str


def load_bundle(bundle_path: str) -> BundleMatrix:
    bundle_path = Path(bundle_path)
    with open(bundle_path) as f:
        bundle = json.load(f)

    observed = bundle["observed_ratings"]
    missing  = bundle["missing_ratings"]
    domain3_meta = bundle.get("domain3_metadata", {})
    is_domain3_item_transductive = (
        domain3_meta.get("experiment_axis") == "item"
        and domain3_meta.get("protocol") == "transductive"
    )

    # ── Discover context vs target annotators from the bundle ─────────────────
    context_annotators = {r["annotator"] for r in observed if r["instance"] == "test"}
    target_annotators  = {r["annotator"] for r in missing  if r["instance"] == "test"}

    # ── Discover all dimensions ───────────────────────────────────────────────
    all_ratings  = observed + missing
    all_anns     = sorted({r["annotator"] for r in all_ratings})
    all_attrs    = sorted({r["attribute"] for r in all_ratings})
    J = len(all_anns)
    I = len(all_attrs)

    # Read C from configs.json if present, else infer from max value
    configs_path = bundle_path.parent / "configs.json"
    if configs_path.exists():
        with open(configs_path) as f:
            cfg = json.load(f)
        C = cfg.get("datagen", cfg).get("C", None)
    else:
        C = None
    if C is None:
        C = max(r["value"] for r in all_ratings)

    # ── Column map ────────────────────────────────────────────────────────────
    ann_to_idx  = {a: i for i, a in enumerate(all_anns)}
    attr_to_idx = {a: i for i, a in enumerate(all_attrs)}
    col_map: Dict[Tuple[int, int], int] = {
        (ann, attr): ann_to_idx[ann] * I + attr_to_idx[attr]
        for ann in all_anns for attr in all_attrs
    }
    D = J * I

    # ── Context / target column masks ─────────────────────────────────────────
    context_cols_mask = np.zeros(D, dtype=bool)
    target_cols_mask  = np.zeros(D, dtype=bool)
    for ann in all_anns:
        for attr in all_attrs:
            col = col_map[(ann, attr)]
            if ann in context_annotators:
                context_cols_mask[col] = True
            elif ann in target_annotators:
                target_cols_mask[col] = True

    # ── Split items by instance ───────────────────────────────────────────────
    item_instance: Dict[int, str] = {}
    for r in all_ratings:
        # An item that appears in multiple instances gets the most informative label.
        # In practice, items are exclusive across train/val/test for these bundles.
        item_instance[r["item"]] = r["instance"]

    train_items = sorted(k for k, v in item_instance.items() if v == "train")
    test_items  = sorted(k for k, v in item_instance.items() if v == "test")

    K_train = len(train_items)
    K_test  = len(test_items)

    train_row = {item: i for i, item in enumerate(train_items)}
    test_row  = {item: i for i, item in enumerate(test_items)}

    # ── Allocate matrices ─────────────────────────────────────────────────────
    X_train = np.full((K_train, D), np.nan, dtype=np.float32)
    M_train = np.zeros((K_train, D), dtype=np.float32)
    X_test  = np.full((K_test,  D), np.nan, dtype=np.float32)
    M_test  = np.zeros((K_test,  D), dtype=np.float32)

    # ── Fill observed ratings ─────────────────────────────────────────────────
    for r in observed:
        item = r["item"]
        col  = col_map[(r["annotator"], r["attribute"])]
        val  = float(r["value"])   # keep 1-indexed
        inst = r["instance"]

        if inst == "train" and item in train_row:
            row = train_row[item]
            X_train[row, col] = val
            M_train[row, col] = 1.0
        elif inst == "test" and item in test_row:
            row = test_row[item]
            X_test[row, col] = val
            M_test[row, col] = 1.0

    # ── Build actual fitting matrix ──────────────────────────────────────────
    if is_domain3_item_transductive:
        fit_items = sorted(train_items + test_items)
        fit_row = {item: i for i, item in enumerate(fit_items)}
        X_fit = np.full((len(fit_items), D), np.nan, dtype=np.float32)
        M_fit = np.zeros((len(fit_items), D), dtype=np.float32)
        for r in observed:
            item = r["item"]
            if item not in fit_row:
                continue
            row = fit_row[item]
            col = col_map[(r["annotator"], r["attribute"])]
            X_fit[row, col] = float(r["value"])
            M_fit[row, col] = 1.0
        training_mode = "transductive_item_domain3"
    else:
        X_fit = X_train
        M_fit = M_train
        fit_items = train_items
        training_mode = "standard"

    # ── Collect ground-truth targets for test evaluation ──────────────────────
    test_targets: List[Tuple[int, int, int]] = []
    for r in missing:
        if r["instance"] != "test":
            continue
        item = r["item"]
        if item not in test_row:
            continue
        row  = test_row[item]
        col  = col_map[(r["annotator"], r["attribute"])]
        val_0idx = r["value"] - 1   # 0-indexed class label
        test_targets.append((row, col, val_0idx))

    meta = {
        "K_train": K_train,
        "K_test":  K_test,
        "J": J,
        "I": I,
        "C": C,
        "D": D,
    }

    return BundleMatrix(
        X_train=X_train,
        M_train=M_train,
        X_test=X_test,
        M_test=M_test,
        X_fit=X_fit,
        M_fit=M_fit,
        col_map=col_map,
        context_annotators=context_annotators,
        target_annotators=target_annotators,
        context_cols_mask=context_cols_mask,
        target_cols_mask=target_cols_mask,
        train_items=train_items,
        test_items=test_items,
        fit_items=fit_items,
        meta=meta,
        test_targets=test_targets,
        training_mode=training_mode,
    )


def compute_metrics(
    probs: np.ndarray,                        # (N, D, C) predicted probabilities
    targets: List[Tuple[int, int, int]],      # (row, col, label_0idx)
) -> dict:
    """Compute accuracy and mean NLL from (N, D, C) probs and target list."""
    eps = 1e-10
    correct = 0
    total_nll = 0.0
    n = len(targets)

    for (row, col, label) in targets:
        p    = probs[row, col, :]
        pred = int(np.argmax(p))
        correct    += int(pred == label)
        total_nll  += -np.log(float(np.clip(p[label], eps, 1.0)))

    return {
        "accuracy": correct / n if n > 0 else float("nan"),
        "mean_nll": total_nll / n if n > 0 else float("nan"),
        "n": n,
    }
