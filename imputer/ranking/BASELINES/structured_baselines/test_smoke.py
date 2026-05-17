#!/usr/bin/env python3
"""
Smoke tests for the new structured factorization.

Covers:
  - SNB probabilities sum to 1 for basic examples
  - CHANGEK fires when sources span two items with same (i, j)
  - CHANGEJ multiplicity (multiple j' annotators with same rating)
  - factor_routing classifies all edge cases correctly
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


def test_routing_change_k() -> None:
    """Source with same (i, j), different k → CHANGE_K."""
    kind = route_source(i_s=0, j_s=0, k_s=1, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.CHANGE_K, f"expected CHANGE_K, got {kind}"


def test_routing_ignored() -> None:
    """Source differing on all three indices → IGNORED."""
    kind = route_source(i_s=1, j_s=1, k_s=1, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.IGNORED, f"expected IGNORED, got {kind}"


def test_routing_cross_item_same_j_only_ignored() -> None:
    """Source with same j only (different i, different k) → IGNORED (not CHANGE_K or ATTR_PAIR)."""
    kind = route_source(i_s=1, j_s=0, k_s=1, i_t=0, j_t=0, k_t=0)
    assert kind == FactorKind.IGNORED, f"expected IGNORED, got {kind}"


def test_multiplicity_change_j() -> None:
    """
    CHANGEJ counter with ratings [3, 3, 4, 5] from four annotators for target (i=0, j=0, k=0).
    Counter should reflect multiplicities: {3: 2, 4: 1, 5: 1}.
    """
    # sources: same i=0, same k=0, different j
    sources = (
        (0, 1, 0, 3),  # j=1, rating=3
        (0, 2, 0, 3),  # j=2, rating=3
        (0, 3, 0, 4),  # j=3, rating=4
        (0, 4, 0, 5),  # j=4, rating=5
    )
    routed = route_sources(sources, i_t=0, j_t=0, k_t=0)
    assert routed.change_j == Counter({3: 2, 4: 1, 5: 1}), f"got {routed.change_j}"
    assert len(routed.attr_pairs) == 0
    assert len(routed.change_k) == 0


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


def test_changek_fires() -> None:
    """
    Two items, same (i=0, j=0) — CHANGEK should fire.

    cells: item 0 and item 1, same attr and annotator.
    Target: (i=0, j=0, k=0, y=1)
    Source: (i=0, j=0, k=1, y=2)  → same (i,j), different k → CHANGE_K
    """
    cells = [(0, 0, 0, 1), (0, 0, 1, 2)]
    # source is the cell on item k=1
    ex = LocalExample(target_i=0, target_j=0, target_k=0, y=1, sources=((0, 0, 1, 2),))
    counts = accumulate_transductive_counts(cells, num_attrs=1, num_classes=3, num_anns=1, num_items=2)
    # Verify counts picked up CHANGE_K evidence
    assert counts.n_change_k.sum() > 0, "n_change_k should have been incremented"
    snb = StructuredNaiveBayes(counts=counts, alpha=1.0)
    probs = snb.predict_proba([ex])
    assert abs(probs[0].sum() - 1.0) < 1e-9, "probs don't sum to 1"


def test_full_transductive_sources() -> None:
    """
    Sources span all splits (train + val + test observed cells).
    Targets from different items should all see cross-item sources.

    Target: (i=0, j=0, k=0)
      - (0, 0, 1, 2): same i=0, j=0, different k=1  → CHANGE_K
      - (1, 0, 0, 0): same j=0, k=0, different i=1  → ATTR_PAIR
    """
    cells = [
        (0, 0, 0, 1),  # target cell: i=0, j=0, k=0
        (0, 0, 1, 2),  # same (i,j), different item k=1   → CHANGE_K
        (1, 0, 0, 0),  # same (j,k), different attr i=1   → ATTR_PAIR
    ]
    # Target: (i=0, j=0, k=0); sources = rest of transductive pool
    all_sources = tuple(c for c in cells if c[:3] != (0, 0, 0))
    ex = LocalExample(target_i=0, target_j=0, target_k=0, y=1, sources=all_sources)
    counts = accumulate_transductive_counts(cells, num_attrs=2, num_classes=3, num_anns=1, num_items=2)
    snb = StructuredNaiveBayes(counts=counts, alpha=1.0)
    probs = snb.predict_proba([ex])
    assert abs(probs[0].sum() - 1.0) < 1e-9

    # Verify routing
    routed = route_sources(all_sources, 0, 0, 0)
    assert 2 in dict(routed.change_k), "CHANGEK source (rating=2) should be present"
    assert len(routed.attr_pairs) > 0, "ATTR_PAIR source should be present"


def main() -> None:
    test_routing_attr_pair()
    test_routing_change_j()
    test_routing_change_k()
    test_routing_ignored()
    test_routing_cross_item_same_j_only_ignored()
    test_multiplicity_change_j()
    test_snb_proba_sum_to_one()
    test_changek_fires()
    test_full_transductive_sources()
    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
