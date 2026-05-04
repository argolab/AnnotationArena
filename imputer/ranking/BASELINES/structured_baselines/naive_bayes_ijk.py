"""
Classic transductive Naive Bayes over (attribute i, annotator j, item k) given class y:

    P(y | i,j,k) ∝ P(y) P(i|y) P(j|y) P(k|y)

with add-one smoothing (same spirit as scripts/utils/plot_llm_rubric_new_stan_curve.py).

This is a *joint-slot* factorization (not the structured relation-aware baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .dataset_adapter import LocalExample, ratings_for_ijk_fit


@dataclass
class NaiveBayesIJK:
    """Categorical NB with independent i,j,k given y."""

    num_classes: int
    num_attrs: int
    num_anns: int
    num_items: int
    class_counts: np.ndarray  # (C,)
    i_counts: np.ndarray  # (C, I)
    j_counts: np.ndarray  # (C, J)
    k_counts: np.ndarray  # (C, K)
    alpha: float = 1.0

    @classmethod
    def fit_from_bundle(
        cls,
        bundle: dict,
        *,
        transductive: bool = True,
        alpha: float = 1.0,
    ) -> "NaiveBayesIJK":
        rows = ratings_for_ijk_fit(bundle, transductive=transductive)
        return cls.fit_from_ratings(rows, alpha=alpha)

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
            ii = int(r["attribute"]) - 1
            jj = int(r["annotator"]) - 1
            kk = int(r["item"]) - 1
            class_counts[y] += 1.0
            i_counts[y, ii] += 1.0
            j_counts[y, jj] += 1.0
            k_counts[y, kk] += 1.0
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
        """Log P(y|i,j,k) for y=0..C-1, shape (C,)."""
        c = self.num_classes
        a = self.alpha
        n = float(self.class_counts.sum())
        log_py = np.log((self.class_counts + a) / (n + a * c))
        log_pi = np.zeros((c, self.num_attrs))
        log_pj = np.zeros((c, self.num_anns))
        log_pk = np.zeros((c, self.num_items))
        for y in range(c):
            denom_i = self.class_counts[y] + a * self.num_attrs
            denom_j = self.class_counts[y] + a * self.num_anns
            denom_k = self.class_counts[y] + a * self.num_items
            log_pi[y] = np.log((self.i_counts[y] + a) / denom_i)
            log_pj[y] = np.log((self.j_counts[y] + a) / denom_j)
            log_pk[y] = np.log((self.k_counts[y] + a) / denom_k)
        scores = log_py + log_pi[:, i] + log_pj[:, j] + log_pk[:, k]
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        return scores - log_norm

    def predict_proba(self, examples: Sequence[LocalExample]) -> np.ndarray:
        out = np.zeros((len(examples), self.num_classes), dtype=np.float64)
        for t, ex in enumerate(examples):
            out[t] = np.exp(self.log_proba_row(ex.target_i, ex.target_j, ex.target_k))
        return out

    def predict(self, examples: Sequence[LocalExample]) -> np.ndarray:
        return np.argmax(self.predict_proba(examples), axis=1)

    def evaluate(self, examples: Sequence[LocalExample]) -> Dict[str, float]:
        probs = self.predict_proba(examples)
        y = np.array([ex.y for ex in examples], dtype=np.int64)
        pred = probs.argmax(axis=1)
        acc = float((pred == y).mean()) if len(y) else float("nan")
        nll = float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean()) if len(y) else float("nan")
        return {"accuracy": acc, "mean_nll": nll, "n": float(len(y))}
