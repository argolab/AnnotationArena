#!/usr/bin/env python3
"""Tiny synthetic plates: verify SNB + log-linear train and predict without crashing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structured_baselines.dataset_adapter import LocalExample
from structured_baselines.log_linear_structured import StructuredLogLinear
from structured_baselines.naive_bayes_structured import StructuredNaiveBayes


def main() -> None:
    I, C = 2, 3
    # Two cells on same item plate share k=0; LOO gives two examples
    exs = [
        LocalExample(
            target_i=0,
            target_j=0,
            target_k=0,
            y=1,
            sources=((1, 1, 0, 2),),
        ),
        LocalExample(
            target_i=1,
            target_j=1,
            target_k=0,
            y=2,
            sources=((0, 0, 0, 1),),
        ),
    ]
    snb = StructuredNaiveBayes.fit(exs, num_attrs=I, num_classes=C, alpha_prior=1.0, alpha_emit=1.0)
    m = snb.evaluate(exs)
    assert m["n"] == 2.0

    ll = StructuredLogLinear.fit(
        exs,
        num_attrs=I,
        num_classes=C,
        epochs=80,
        lr=0.2,
        batch_size=2,
        device="cpu",
        verbose=False,
    )
    m2 = ll.evaluate(exs)
    assert m2["mean_nll"] < 1.5, m2  # should fit tiny set
    print("smoke OK", {"snb": m, "ll": m2})


if __name__ == "__main__":
    main()
