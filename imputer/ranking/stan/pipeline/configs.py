from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

# -----------------------------------------------------------------------------
# Stan data: all fields on DataGenConfig (see stan/docs/STAN_DATA_EXTRA.md)
# -----------------------------------------------------------------------------

# Core keys always present in Stan data (from config; not type-specific).
CORE_STAN_KEYS: frozenset = frozenset({
    "K_train", "K_test", "I", "J", "C",
    "enable_pairwise_rankings", "pairwise_cap_per_item",
})

# For each stan_type, the exact set of (non-core) Stan data field names that must be set
# on the config (non-None). No other Stan-data field may be set for that type.
# All values are passed from config into Stan data; there is no separate "extra" dict.
STAN_TYPE_REQUIRED: Dict[str, Set[str]] = {
    "discrete": {"M", "S", "sigma_measurement", "kappa", "temperature"},
    "normal-noise-dot-product": {
        "D", "d_annotator", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
        "use_factored_annotator", "derive_thresholds_from_annotator",
    },
    "factored-dot-product": {
        "D", "d_annotator", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
        "use_factored_annotator", "derive_thresholds_from_annotator",
    },
    "tensor": {
        "D", "factor_decay", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
    },
}

# All Stan-data field names that are type-specific (used for validation: only required set may be non-None).
STAN_DATA_FIELDS: frozenset = frozenset({
    "D", "M", "S", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
    "use_factored_annotator", "derive_thresholds_from_annotator", "d_annotator", "factor_decay",
})


def check_config_for_stan_type(config: "DataGenConfig") -> None:
    """
    Validate that for config.stan_type the config has exactly the required set of
    Stan-data fields set (non-None) and all other Stan-data fields are None.
    Raises ValueError if not.
    """
    stan_type = config.stan_type
    if stan_type not in STAN_TYPE_REQUIRED:
        raise ValueError(
            f"Unknown stan_type: {stan_type!r}. Must be one of {list(STAN_TYPE_REQUIRED)}."
        )
    required = STAN_TYPE_REQUIRED[stan_type]
    for key in required:
        val = getattr(config, key, None)
        if val is None:
            raise ValueError(
                f"For stan_type={stan_type!r}, config must set {key!r} (currently None)."
            )
    for key in STAN_DATA_FIELDS:
        if key in required:
            continue
        val = getattr(config, key, None)
        if val is not None:
            raise ValueError(
                f"For stan_type={stan_type!r}, config must not set {key!r} (must be None); got {val!r}."
            )


@dataclass
class DataGenConfig:
    """
    Configuration for Stan data generation and downstream inference.

    Design:
    - Core: K_train, K_test, I, J, C, enable_pairwise_rankings, pairwise_cap_per_item.
    - All other Stan-related fields are explicit attributes, default None. For a given
      stan_type, exactly the set in STAN_TYPE_REQUIRED[stan_type] must be set (non-None);
      all other Stan-data fields must be None (enforced by check_config_for_stan_type).
    - to_stan_data() validates and returns a single Stan data dict (core + type-specific
      fields from this config), used for data generation and for saving stan_data.json.
    """

    # --- Core (always required) ---
    K_train: int
    K_test: int
    I: int
    J: int
    C: int

    enable_third_annotator: bool = True
    enable_pairwise_rankings: bool = True
    pairwise_cap_per_item: int = 10

    # --- Stan-data fields (all optional; exactly the set for stan_type must be set) ---
    D: Optional[int] = None
    M: Optional[int] = None
    S: Optional[int] = None
    sigma_annotator: Optional[float] = None
    sigma_measurement: Optional[float] = None
    kappa: Optional[float] = None
    temperature: Optional[float] = None
    use_factored_annotator: Optional[int] = None  # 0 or 1
    derive_thresholds_from_annotator: Optional[int] = None  # 0 or 1
    d_annotator: Optional[int] = None
    factor_decay: Optional[float] = None

    # Misspecification flags (only used with tensor_data_generation.stan)
    use_log_scores: bool = False       # Apply log() to raw CP scores before binning
    use_logistic_link: bool = False    # Use inv_logit instead of Phi for binning
    use_normal_loadings: bool = False  # Use N(0,1) loadings instead of Exp(1)

    # Observation protocol
    observation_protocol: str = "tie_breaking"  # "tie_breaking", "mcar", "extended_rankings"
    mcar_missing_rate: float = 0.5  # Missing rate for MCAR protocol (was mar_missing_rate)
    pairwise_observation_rate: float = 1.0  # For tie_breaking: fraction of missing pairwise rankings to observe (0.0-1.0)

    seed: Optional[int] = None

    stan_type: str = "factored-dot-product"

    def to_stan_data(self) -> Dict[str, Any]:
        """
        Build the full Stan data dict from this config (core + type-specific fields).
        Validates that for self.stan_type exactly the required fields are set.
        Used for data generation and for saving stan_data.json for inference.
        """
        check_config_for_stan_type(self)

        base: Dict[str, Any] = {
            "K_train": self.K_train,
            "K_test": self.K_test,
            "I": self.I,
            "J": self.J,
            "C": self.C,
            "enable_pairwise_rankings": 1 if self.enable_pairwise_rankings else 0,
            "pairwise_cap_per_item": self.pairwise_cap_per_item,
        }

        required = STAN_TYPE_REQUIRED[self.stan_type]
        for key in sorted(required):
            val = getattr(self, key)
            if key in ("use_factored_annotator", "derive_thresholds_from_annotator"):
                base[key] = int(val) if isinstance(val, (bool, int)) else val
            else:
                base[key] = val

        # Discrete Stan file still expects D and sigma_annotator for compatibility (not used in generation).
        if self.stan_type == "discrete":
            base.setdefault("D", 1)
            base.setdefault("sigma_annotator", 0.1)
        # Tensor CP model: all dimensions match (rank D); d_annotator is set to D for Stan data block compatibility.
        if self.stan_type == "tensor":
            base["d_annotator"] = base["D"]

        return base


@dataclass
class DomainConfig:
    K: int
    I: int
    J: int
    D: int
    C: int

    ranking_size: int = 2
    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    kappa: float = 2.0
    temperature: float = 0.5
    sigma_embedding_prior: float = 1.0
    sigma_preference_prior: float = 1.0


@dataclass
class McmcConfig:
    chains: int = 4
    iter_warmup: int = 1000
    iter_sampling: int = 2000
    adapt_delta: float = 0.8
    max_treedepth: int = 15
    seed: Optional[int] = 42
