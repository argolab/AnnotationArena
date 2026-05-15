"""Fit and evaluate the three structured baselines on one bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .cli_defaults import DEFAULT_IJK_ALPHA, DEFAULT_SNB_ALPHA, DEFAULT_UNIGRAM_ALPHA
from .dataset_adapter import (
    build_eval_examples,
    build_test_examples,
    bundle_dims,
    load_bundle_dict,
)
from .naive_bayes_ijk import NaiveBayesIJK
from .naive_bayes_structured import StructuredNaiveBayes
from .unigram_pooled import PooledUnigramIJ


@dataclass
class FittedBaselines:
    unigram_ij: PooledUnigramIJ
    nb_ijk: NaiveBayesIJK
    snb: StructuredNaiveBayes


def fit_baselines(
    bundle: dict,
    bundle_path: Path | None = None,
    *,
    unigram_alpha: float = DEFAULT_UNIGRAM_ALPHA,
    ijk_alpha: float = DEFAULT_IJK_ALPHA,
    snb_alpha: float = DEFAULT_SNB_ALPHA,
) -> FittedBaselines:
    I, J, C = bundle_dims(bundle, bundle_path)
    K = max(
        int(r["item"])
        for r in (bundle.get("observed_ratings", []) + bundle.get("missing_ratings", []))
    )
    return FittedBaselines(
        unigram_ij=PooledUnigramIJ.fit(bundle, alpha=unigram_alpha),
        nb_ijk=NaiveBayesIJK.fit_from_bundle(bundle, alpha=ijk_alpha),
        snb=StructuredNaiveBayes.fit_from_bundle(
            bundle, num_attrs=I, num_classes=C, num_anns=J, num_items=K, alpha=snb_alpha
        ),
    )


def evaluate_split(
    fitted: FittedBaselines,
    bundle: dict,
    split: Literal["test", "val"] = "test",
) -> dict[str, dict]:
    if split == "test":
        ex = build_test_examples(bundle)
    else:
        ex = build_eval_examples(bundle, split)
    return {
        "unigram_ij": fitted.unigram_ij.evaluate_split(bundle, split),
        "ijk": fitted.nb_ijk.evaluate(ex),
        "snb": fitted.snb.evaluate(ex),
    }


def load_and_fit(bundle_path: Path, **kwargs) -> tuple[dict, FittedBaselines]:
    bundle = load_bundle_dict(bundle_path)
    return bundle, fit_baselines(bundle, bundle_path, **kwargs)


def calibration_probs_labels(
    fitted: FittedBaselines,
    bundle: dict,
    split: Literal["test", "val"] = "test",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Per-model (probs, labels) on missing cells for reliability diagrams.

    ``probs`` shape (n, C), ``labels`` 0-based class indices.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing = [r for r in bundle.get("missing_ratings", []) if str(r.get("instance")) == split]
    if not missing:
        return out

    probs_u = np.stack([fitted.unigram_ij.proba_for_row(r) for r in missing], axis=0)
    labels_u = np.asarray([int(r["value"]) - 1 for r in missing], dtype=np.int64)
    out["unigram_ij"] = (probs_u, labels_u)

    if split == "test":
        ex = build_test_examples(bundle)
    else:
        ex = build_eval_examples(bundle, split)
    if ex:
        labels = np.asarray([ex.y for ex in ex], dtype=np.int64)
        out["ijk"] = (fitted.nb_ijk.predict_proba(ex), labels)
        out["snb"] = (fitted.snb.predict_proba(ex), labels)
    return out
