"""Sample core unroll depth during training (Huginn-style log-normal Poisson or uniform)."""

from __future__ import annotations

import math
import random
import warnings
from typing import Optional

RECURRENCE_DISTRIBUTIONS = ("lognormal_poisson", "uniform")


def validate_recurrence_bounds(min_r: int, max_r: int, mean_r: int) -> None:
    if min_r < 1:
        raise ValueError(f"min_r must be >= 1, got {min_r}")
    if min_r > max_r:
        raise ValueError(f"min_r ({min_r}) must be <= max_r ({max_r})")
    if mean_r > max_r:
        warnings.warn(
            f"mean_r ({mean_r}) > max_r ({max_r}); samples will be clipped to max_r.",
            stacklevel=2,
        )


def _sample_poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0.0:
        return 0
    limit = math.exp(-lam)
    k = 0
    product = 1.0
    while product > limit:
        k += 1
        product *= rng.random()
    return k - 1


def sample_recurrence(
    rng: Optional[random.Random],
    *,
    mean_r: int,
    min_r: int,
    max_r: int,
    distribution: str = "lognormal_poisson",
    sigma: float = 0.5,
) -> int:
    """
    Draw one recurrence count for a training step.

    lognormal_poisson (Huginn §3.3): tau ~ N(log(mean_r) - sigma^2/2, sigma),
    r = Poisson(exp(tau)) + 1, then clip to [min_r, max_r].
    """
    validate_recurrence_bounds(min_r, max_r, mean_r)
    if distribution not in RECURRENCE_DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {RECURRENCE_DISTRIBUTIONS}, got {distribution!r}"
        )
    if rng is None:
        rng = random.Random()

    if distribution == "uniform":
        return rng.randint(min_r, max_r)

    mu = math.log(float(mean_r)) - (sigma**2) / 2.0
    tau = rng.gauss(mu, sigma)
    rate = math.exp(tau)
    r = _sample_poisson(rng, rate) + 1
    return max(min_r, min(max_r, int(r)))
