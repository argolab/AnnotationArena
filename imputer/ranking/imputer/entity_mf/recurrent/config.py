from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecurrentMarformerConfig:
    """
    Configuration for Recurrent Entity Marformer.

    Depth knobs:
      - prelude_depth: unique blocks before recurrence
      - num_core_layers: distinct parameter layers in the recurrent core
      - num_recurrence: how many times the core stack is applied
      - coda_depth: unique blocks after recurrence

    effective_depth = prelude_depth + num_core_layers * num_recurrence + coda_depth
    """

    embedding_dim: int = 72
    attention_heads: int = 4
    dropout: float = 0.1
    d_ff: int = 128
    num_ffn_layers: int = 1
    logit_high: float = 20.0
    temperature: float = 1.0
    use_per_head_rel: bool = True
    use_pointer: bool = False
    use_rel_value: bool = False
    use_addone_attn: bool = False
    type_embedding_init: str = "normal"
    use_deviation_norm: bool = False
    scale_shared_rel: bool = False
    use_graph_mask: bool = False
    use_param_output_head: bool = False

    prelude_depth: int = 1
    num_core_layers: int = 2
    num_recurrence: int = 3
    coda_depth: int = 1

    randomize_recurrence: bool = False
    recurrence_min: int = 1
    recurrence_max: int | None = None
    recurrence_distribution: str = "lognormal_poisson"
    recurrence_sigma: float = 0.5

    # Deep supervision: always unroll to num_recurrence; auxiliary CE on each exit.
    # Requires coda_depth=0 (no coda head).
    deep_supervision: bool = False
    deep_supervision_schedule: str = "exp_decay"
    deep_supervision_exp_base: float = 1.12

    def validate(self) -> None:
        for name, value in (
            ("prelude_depth", self.prelude_depth),
            ("num_core_layers", self.num_core_layers),
            ("num_recurrence", self.num_recurrence),
            ("coda_depth", self.coda_depth),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.effective_depth < 1:
            raise ValueError(
                "Total depth must be >= 1: "
                f"prelude_depth + num_core_layers * num_recurrence + coda_depth "
                f"= {self.effective_depth}"
            )
        if self.recurrence_max is None:
            self.recurrence_max = self.num_recurrence
        if self.recurrence_min < 1:
            raise ValueError(f"recurrence_min must be >= 1, got {self.recurrence_min}")
        if self.recurrence_min > self.recurrence_max:
            raise ValueError(
                f"recurrence_min ({self.recurrence_min}) must be <= "
                f"recurrence_max ({self.recurrence_max})"
            )
        from .recurrence_schedule import RECURRENCE_DISTRIBUTIONS

        if self.recurrence_distribution not in RECURRENCE_DISTRIBUTIONS:
            raise ValueError(
                f"recurrence_distribution must be one of {RECURRENCE_DISTRIBUTIONS}, "
                f"got {self.recurrence_distribution!r}"
            )
        if self.recurrence_sigma <= 0:
            raise ValueError(f"recurrence_sigma must be > 0, got {self.recurrence_sigma}")
        from .deep_supervision import DEEP_SUPERVISION_SCHEDULES

        if self.deep_supervision_schedule not in DEEP_SUPERVISION_SCHEDULES:
            raise ValueError(
                f"deep_supervision_schedule must be one of {DEEP_SUPERVISION_SCHEDULES}, "
                f"got {self.deep_supervision_schedule!r}"
            )
        if self.deep_supervision and self.randomize_recurrence:
            raise ValueError(
                "deep_supervision and randomize_recurrence are mutually exclusive"
            )
        if self.deep_supervision and self.coda_depth != 0:
            raise ValueError(
                f"deep_supervision requires coda_depth=0, got {self.coda_depth}"
            )
        if self.deep_supervision_exp_base <= 1.0:
            raise ValueError(
                f"deep_supervision_exp_base must be > 1.0, got {self.deep_supervision_exp_base}"
            )

    @property
    def effective_depth(self) -> int:
        return self.prelude_depth + self.num_core_layers * self.num_recurrence + self.coda_depth
