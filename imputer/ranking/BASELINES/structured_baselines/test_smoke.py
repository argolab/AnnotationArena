#!/usr/bin/env python3
"""
Smoke tests for the structured factorization (ATTR_PAIR + CHANGEJ only).

Covers:
  - SNB probabilities sum to 1 for basic examples
  - CHANGEJ multiplicity (multiple j' annotators with same rating)
  - factor_routing classifies all edge cases correctly
  - Same (i, j), different k is IGNORED (no CHANGEK factor)
  - Sources are taken from the full transductive pool (not same-item only)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structured_baselines.dataset_adapter import LocalExample
from structured_baselines.factor_routing import FactorKind, route_source, route_sources
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes
from structured_baselines.plate_graph_factorized import accumulate_transductive_counts


# ---------------------------------------------------------------------------
# Routing unit tests
# ---------------------------------------------------------------------------

def test_routing_attr_pair() -> None:
    """Source with same (j, k), different i → ATTR_PAIR."""
    kind = route_source(i_s=1, j_s=0, k_s=0, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.ATTR_PAIR, f"expected ATTR_PAIR, got {kind}"


def test_routing_change_j() -> None:
    """Source with same (i, k), different j → CHANGE_J."""
    kind = route_source(i_s=0, j_s=1, k_s=0, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.CHANGE_J, f"expected CHANGE_J, got {kind}"


def test_routing_same_ij_diff_k_ignored() -> None:
    """Source with same (i, j), different k → IGNORED."""
    kind = route_source(i_s=0, j_s=0, k_s=1, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.IGNORED, f"expected IGNORED, got {kind}"


def test_routing_ignored() -> None:
    """Source differing on all three indices → IGNORED."""
    kind = route_source(i_s=1, j_s=1, k_s=1, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.IGNORED, f"expected IGNORED, got {kind}"


def test_routing_cross_item_same_j_only_ignored() -> None:
    """Source with same j only (different i, different k) → IGNORED."""
    kind = route_source(i_s=1, j_s=0, k_s=1, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.IGNORED, f"expected IGNORED, got {kind}"


def test_multiplicity_change_j() -> None:
    """
    CHANGEJ counter with ratings [3, 3, 4, 5] from four annotators for target (i=0, j=0, k=0).
    Counter should reflect multiplicities: {3: 2, 4: 1, 5: 1}.
    """
    sources = (
        (0, 1, 0, 3),
        (0, 2, 0, 3),
        (0, 3, 0, 4),
        (0, 4, 0, 5),
    )
    routed = route_sources(sources, i_t=0, j_t=0, k_t=0)
    assert routed.change_j == Counter({3: 2, 4: 1, 5: 1}), f"got {routed.change_j}"
    assert len(routed.attr_pairs) == 0


# ---------------------------------------------------------------------------
# SNB integration tests
# ---------------------------------------------------------------------------

def test_snb_proba_sum_to_one() -> None:
    """Basic SNB with same-item attr-pair sources."""
    cells = [(0, 0, 0, 1), (1, 1, 0, 2)]
    exs = [
        LocalExample(0, 0, 0, 1, ((1, 1, 0, 2),)),
        LocalExample(1, 1, 0, 2, ((0, 0, 0, 1),)),
    ]
    counts = accumulate_transductive_counts(cells, num_attrs=2, num_classes=3)
    snb = StructuredNaiveBayes(counts=counts, alpha=1.0)
    m = snb.evaluate(exs)
    assert m["n"] == 2.0
    probs = snb.predict_proba(exs)
    for row in probs:
        assert abs(row.sum() - 1.0) < 1e-9, f"probs don't sum to 1: {row}"


def test_same_ij_diff_k_not_counted() -> None:
    """Cross-item source with same (i, j) does not increment pairwise counts."""
    cells = [(0, 0, 0, 1), (0, 0, 1, 2)]
    counts = accumulate_transductive_counts(cells, num_attrs=1, num_classes=3, num_anns=1, num_items=2)
    assert counts.n_change_j.sum() == 0
    # Only ordered pairs that route to ATTR_PAIR or CHANGE_J increment n_attr / n_change_j
    assert counts.n_attr.sum() == 0


def test_full_transductive_sources() -> None:
    """
    Sources span all splits (train + val + test observed cells).
    Target (i=0, j=0, k=0) should see ATTR_PAIR from (1, 0, 0);
    same (i, j), different k is ignored.
    """
    cells = [
        (0, 0, 0, 1),
        (0, 0, 1, 2),
        (1, 0, 0, 0),
    ]
    all_sources = tuple(c for c in cells if c[:3] != (0, 0, 0))
    ex = LocalExample(target_i=0, target_j=0, target_k=0, y=1, sources=all_sources)
    counts = accumulate_transductive_counts(cells, num_attrs=2, num_classes=3, num_anns=1, num_items=2)
    snb = StructuredNaiveBayes(counts=counts, alpha=1.0)
    probs = snb.predict_proba([ex])
    assert abs(probs[0].sum() - 1.0) < 1e-9

    routed = route_sources(all_sources, 0, 0, 0)
    assert len(routed.attr_pairs) > 0, "ATTR_PAIR source should be present"
    assert route_source(0, 0, 1, 0, 0, 0) == FactorKind.IGNORED


def main() -> None:
    test_routing_attr_pair()
    test_routing_change_j()
    test_routing_same_ij_diff_k_ignored()
    test_routing_ignored()
    test_routing_cross_item_same_j_only_ignored()
    test_multiplicity_change_j()
    test_snb_proba_sum_to_one()
    test_same_ij_diff_k_not_counted()
    test_full_transductive_sources()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
