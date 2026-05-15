"""Pooled unigram P(y | i, j) on the transductive observed pool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .dataset_adapter import transductive_observed_rows

PoolKey = Tuple[int, int]


@dataclass
class PooledUnigramIJ:
    num_classes: int
    alpha: float
    pool_counts: Dict[PoolKey, List[float]]

    @classmethod
    def fit(cls, bundle: dict, *, alpha: float = 1.0) -> "PooledUnigramIJ":
        observed = transductive_observed_rows(bundle)
        if not observed:
            raise ValueError("No transductive observed ratings in bundle.")
        missing = bundle.get("missing_ratings", [])
        c = max(int(r["value"]) for r in (observed + missing))
        pool: Dict[PoolKey, List[float]] = {}
        for row in observed:
            key = (int(row["attribute"]), int(row["annotator"]))
            if key not in pool:
                pool[key] = [0.0] * c
            idx = int(row["value"]) - 1
            if 0 <= idx < c:
                pool[key][idx] += 1.0
        return cls(num_classes=c, alpha=float(alpha), pool_counts=pool)

    def proba_for_row(self, row: dict) -> np.ndarray:
        c = self.num_classes
        key = (int(row["attribute"]), int(row["annotator"]))
        counts = self.pool_counts.get(key, [0.0] * c)
        denom = sum(counts) + self.alpha * c
        return np.asarray([(counts[ii] + self.alpha) / denom for ii in range(c)], dtype=np.float64)

    def evaluate_split(self, bundle: dict, instance: str = "test") -> Dict[str, float]:
        missing = [r for r in bundle.get("missing_ratings", []) if str(r.get("instance")) == instance]
        if not missing:
            return {"accuracy": float("nan"), "mean_nll": float("nan"), "rmse": float("nan"), "n": 0.0}
        nll = 0.0
        se = 0.0
        correct = 0
        classes = np.arange(1, self.num_classes + 1, dtype=np.float64)
        for row in missing:
            probs = self.proba_for_row(row)
            y = int(row["value"]) - 1
            nll -= math.log(float(probs[y]) + 1e-12)
            if int(probs.argmax()) == y:
                correct += 1
            se += (float(probs @ classes) - float(int(row["value"]))) ** 2
        n = len(missing)
        return {
            "accuracy": correct / n,
            "mean_nll": nll / n,
            "rmse": math.sqrt(se / n),
            "n": float(n),
        }
