"""Structured NB: global transductive pool + factorized pairwise terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .dataset_adapter import LocalExample, transductive_observed_cells
from .plate_graph_factorized import (
    FactorizedPlateCounts,
    StructuredFactorMask,
    accumulate_transductive_counts,
    log_proba_normalized,
)


@dataclass
class StructuredNaiveBayes:
    counts: FactorizedPlateCounts
    alpha: float
    factor_mask: StructuredFactorMask = StructuredFactorMask.all_on()

    @classmethod
    def fit_from_bundle(
        cls,
        bundle: dict,
        num_attrs: int,
        num_classes: int,
        *,
        num_anns: int | None = None,
        num_items: int | None = None,
        alpha: float = 1.0,
        factor_mask: StructuredFactorMask | None = None,
    ) -> "StructuredNaiveBayes":
        mask = factor_mask if factor_mask is not None else StructuredFactorMask.all_on()
        cells = transductive_observed_cells(bundle)
        counts = accumulate_transductive_counts(
            cells,
            num_attrs=num_attrs,
            num_classes=num_classes,
            num_anns=num_anns,
            num_items=num_items,
            factor_mask=mask,
        )
        return cls(counts=counts, alpha=float(alpha), factor_mask=mask)

    def log_proba_one(self, ex: LocalExample) -> np.ndarray:
        return log_proba_normalized(
            self.counts, ex, self.alpha, factor_mask=self.factor_mask
        )

    def predict_proba(self, examples: Sequence[LocalExample]) -> np.ndarray:
        out = np.zeros((len(examples), self.counts.num_classes), dtype=np.float64)
        for t, ex in enumerate(examples):
            out[t] = np.exp(self.log_proba_one(ex))
        return out

    def evaluate(self, examples: Sequence[LocalExample]) -> Dict[str, float]:
        probs = self.predict_proba(examples)
        y = np.array([ex.y for ex in examples], dtype=np.int64)
        pred = probs.argmax(axis=1)
        acc = float((pred == y).mean()) if len(y) else float("nan")
        nll = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()) if len(y) else float("nan")
        return {"accuracy": acc, "mean_nll": nll, "n": float(len(y))}
