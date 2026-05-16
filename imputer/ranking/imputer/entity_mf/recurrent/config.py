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

    @property
    def effective_depth(self) -> int:
        return self.prelude_depth + self.num_core_layers * self.num_recurrence + self.coda_depth
