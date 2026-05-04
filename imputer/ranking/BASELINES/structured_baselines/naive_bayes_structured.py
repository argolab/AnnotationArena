"""
Relation-aware conditional Naive Bayes for predicting one masked cell from sources.

This version includes target/source (i,j,k) ids directly:

    score(y) =
        log P(Y=y | i_t, j_t, k_t)
        + sum_r log P(i_s, j_s, k_s, v_s | i_t, j_t, k_t, Y=y, rel_r)

where rel_r is produced by `feature_utils.relation_label` and is conditioned (not modeled
as jointly drawn with the source cell).

Counts are stored sparsely in dictionaries. Add-alpha smoothing:
  prior bins per target context: C
  per (target context, y, rel), emission is a multinomial over I * J * K * C source cells
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .dataset_adapter import LocalExample
from .feature_utils import relation_label


@dataclass
class StructuredNaiveBayes:
    num_attrs: int
    num_anns: int
    num_items: int
    num_classes: int
    prior_counts: Dict[Tuple[int, int, int, int], float]  # (i_t,j_t,k_t,y) -> count
    prior_totals: Dict[Tuple[int, int, int], float]  # (i_t,j_t,k_t) -> total count
    emit_counts: Dict[Tuple[int, int, int, int, int, int, int, int, int], float]
    # key: (i_t,j_t,k_t,y, rel, i_s,j_s,k_s,v_s) -> count
    emit_totals_by_rel: Dict[Tuple[int, int, int, int, int], float]
    # (i_t,j_t,k_t,y, rel) -> total count of sources in that relation bucket
    alpha_prior: float
    alpha_emit: float

    @classmethod
    def fit(
        cls,
        examples: Sequence[LocalExample],
        num_attrs: int,
        num_classes: int,
        num_anns: int | None = None,
        num_items: int | None = None,
        *,
        alpha_prior: float = 1.0,
        alpha_emit: float = 1.0,
    ) -> "StructuredNaiveBayes":
        I, C = num_attrs, num_classes
        if num_anns is None or num_items is None:
            max_j = -1
            max_k = -1
            for ex in examples:
                max_j = max(max_j, ex.target_j)
                max_k = max(max_k, ex.target_k)
                for (_i_s, j_s, k_s, _v_s) in ex.sources:
                    max_j = max(max_j, j_s)
                    max_k = max(max_k, k_s)
            if num_anns is None:
                num_anns = max_j + 1
            if num_items is None:
                num_items = max_k + 1

        prior: Dict[Tuple[int, int, int, int], float] = {}
        prior_totals: Dict[Tuple[int, int, int], float] = {}
        emit: Dict[Tuple[int, int, int, int, int, int, int, int, int], float] = {}
        emit_totals_by_rel: Dict[Tuple[int, int, int, int, int], float] = {}
        for ex in examples:
            it = ex.target_i
            jt = ex.target_j
            kt = ex.target_k
            y = ex.y
            pk = (it, jt, kt, y)
            ptk = (it, jt, kt)
            prior[pk] = prior.get(pk, 0.0) + 1.0
            prior_totals[ptk] = prior_totals.get(ptk, 0.0) + 1.0
            for (i_s, j_s, k_s, v_s) in ex.sources:
                rel = relation_label(i_s, j_s, k_s, it, jt, kt)
                ek = (it, jt, kt, y, rel, i_s, j_s, k_s, v_s)
                etr = (it, jt, kt, y, rel)
                emit[ek] = emit.get(ek, 0.0) + 1.0
                emit_totals_by_rel[etr] = emit_totals_by_rel.get(etr, 0.0) + 1.0
        return cls(
            num_attrs=I,
            num_anns=int(num_anns),
            num_items=int(num_items),
            num_classes=C,
            prior_counts=prior,
            prior_totals=prior_totals,
            emit_counts=emit,
            emit_totals_by_rel=emit_totals_by_rel,
            alpha_prior=alpha_prior,
            alpha_emit=alpha_emit,
        )

    def log_proba_one(self, ex: LocalExample) -> np.ndarray:
        """Log P(y | sources) for y=0..C-1."""
        it, jt, kt = ex.target_i, ex.target_j, ex.target_k
        scores = np.zeros(self.num_classes, dtype=np.float64)
        prior_total = self.prior_totals.get((it, jt, kt), 0.0)
        prior_denom = prior_total + self.alpha_prior * self.num_classes
        emit_vocab = self.num_attrs * self.num_anns * self.num_items * self.num_classes
        for y in range(self.num_classes):
            pcount = self.prior_counts.get((it, jt, kt, y), 0.0)
            s = math.log((pcount + self.alpha_prior) / prior_denom)
            for (i_s, j_s, k_s, v_s) in ex.sources:
                rel = relation_label(i_s, j_s, k_s, it, jt, kt)
                etr = (it, jt, kt, y, rel)
                total_rel = self.emit_totals_by_rel.get(etr, 0.0)
                denom = total_rel + self.alpha_emit * emit_vocab
                cnt = self.emit_counts.get((it, jt, kt, y, rel, i_s, j_s, k_s, v_s), 0.0)
                s += math.log((cnt + self.alpha_emit) / denom)
            scores[y] = s
        m = float(scores.max())
        log_norm = m + math.log(float(np.sum(np.exp(scores - m))))
        return scores - log_norm

    def predict_proba(self, examples: Sequence[LocalExample]) -> np.ndarray:
        out = np.zeros((len(examples), self.num_classes), dtype=np.float64)
        for t, ex in enumerate(examples):
            out[t] = np.exp(self.log_proba_one(ex))
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
