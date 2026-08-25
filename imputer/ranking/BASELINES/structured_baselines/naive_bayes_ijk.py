"""P(y | i,j,k) ∝ P(y) P(i|y) P(j|y) P(k|y) on the transductive observed pool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .dataset_adapter import LocalExample, transductive_observed_rows


@dataclass
class NaiveBayesIJK:
    num_classes: int
    num_attrs: int
    num_anns: int
    num_items: int
    class_counts: np.ndarray
    i_counts: np.ndarray
    j_counts: np.ndarray
    k_counts: np.ndarray
    alpha: float = 1.0

    @classmethod
    def fit_from_bundle(cls, bundle: dict, *, alpha: float = 1.0) -> "NaiveBayesIJK":
        return cls.fit_from_ratings(transductive_observed_rows(bundle), alpha=alpha)

    @classmethod
    def fit_from_ratings(cls, rows: Sequence[dict], *, alpha: float = 1.0) -> "NaiveBayesIJK":
        c = max(int(r["value"]) for r in rows)
        max_i = max(int(r["attribute"]) for r in rows)
        max_j = max(int(r["annotator"]) for r in rows)
        max_k = max(int(r["item"]) for r in rows)
        class_counts = np.zeros(c, dtype=np.float64)
        i_counts = np.zeros((c, max_i), dtype=np.float64)
        j_counts = np.zeros((c, max_j), dtype=np.float64)
        k_counts = np.zeros((c, max_k), dtype=np.float64)
        for r in rows:
            y = int(r["value"]) - 1
            class_counts[y] += 1.0
            i_counts[y, int(r["attribute"]) - 1] += 1.0
            j_counts[y, int(r["annotator"]) - 1] += 1.0
            k_counts[y, int(r["item"]) - 1] += 1.0
        return cls(
            num_classes=c,
            num_attrs=max_i,
            num_anns=max_j,
            num_items=max_k,
            class_counts=class_counts,
            i_counts=i_counts,
            j_counts=j_counts,
            k_counts=k_counts,
            alpha=alpha,
        )

    def log_proba_row(self, i: int, j: int, k: int) -> np.ndarray:
        c = self.num_classes
        a = self.alpha
        n = float(self.class_counts.sum())
        log_py = np.log((self.class_counts + a) / (n + a * c))
        scores = np.zeros(c, dtype=np.float64)
        for y in range(c):
            scores[y] = (
                log_py[y]
                + np.log((self.i_counts[y, i] + a) / (self.class_counts[y] + a * self.num_attrs))
                + np.log((self.j_counts[y, j] + a) / (self.class_counts[y] + a * self.num_anns))
                + np.log((self.k_counts[y, k] + a) / (self.class_counts[y] + a * self.num_items))
            )
        m = float(scores.max())
        return scores - (m + math.log(float(np.sum(np.exp(scores - m)))))

    def predict_proba(self, examples: Sequence[LocalExample]) -> np.ndarray:
        out = np.zeros((len(examples), self.num_classes), dtype=np.float64)
        for t, ex in enumerate(examples):
            out[t] = np.exp(self.log_proba_row(ex.target_i, ex.target_j, ex.target_k))
        return out

    def evaluate(self, examples: Sequence[LocalExample]) -> Dict[str, float]:
        probs = self.predict_proba(examples)
        y = np.array([ex.y for ex in examples], dtype=np.int64)
        pred = probs.argmax(axis=1)
        acc = float((pred == y).mean()) if len(y) else float("nan")
        nll = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()) if len(y) else float("nan")
        return {"accuracy": acc, "mean_nll": nll, "n": float(len(y))}
