"""Auxiliary-head loss weights for recurrent deep supervision (coda_depth=0)."""

from __future__ import annotations

DEEP_SUPERVISION_SCHEDULES = ("exp_decay", "linear", "uniform")
DEFAULT_EXP_BASE = 1.12  # gentle growth shallow→deep; ~5.5× at r=16 vs ~16× for linear


def deep_supervision_weights(
    num_recurrence: int,
    *,
    schedule: str = "exp_decay",
    exp_base: float = DEFAULT_EXP_BASE,
) -> list[float]:
    """
    Weights for auxiliary losses after each core unroll (k=1..num_recurrence).

    Shallower exits receive weaker weight; deeper exits receive stronger weight.
    ``exp_decay``: w_k ∝ exp_base^(k-1) (normalized). Shallow heads retain more
    mass than linear when exp_base is near 1.
    """
    if num_recurrence < 1:
        raise ValueError(f"num_recurrence must be >= 1, got {num_recurrence}")
    if schedule not in DEEP_SUPERVISION_SCHEDULES:
        raise ValueError(
            f"schedule must be one of {DEEP_SUPERVISION_SCHEDULES}, got {schedule!r}"
        )
    if exp_base <= 0.0:
        raise ValueError(f"exp_base must be > 0, got {exp_base}")

    if schedule == "uniform":
        raw = [1.0] * num_recurrence
    elif schedule == "linear":
        raw = [float(k) / float(num_recurrence) for k in range(1, num_recurrence + 1)]
    else:
        raw = [exp_base ** (k - 1) for k in range(1, num_recurrence + 1)]

    total = sum(raw)
    return [w / total for w in raw]
