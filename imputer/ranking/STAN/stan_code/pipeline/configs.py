from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

# -----------------------------------------------------------------------------
# Stan data: all fields on DataGenConfig (see stan/docs/STAN_DATA_EXTRA.md)
# -----------------------------------------------------------------------------

# Core keys always present in Stan data (from config; not type-specific).
CORE_STAN_KEYS: frozenset = frozenset({
    "K_train", "K_test", "K_val", "I", "J", "C",
    "enable_pairwise_rankings", "pairwise_cap_per_item",
})

# For each stan_type, the exact set of (non-core) Stan data field names that must be
# set on the config (non-None). No other Stan-data field may be set for that type.
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
    "dawid-skene": {
        "D", "d_annotator", "sigma_annotator", "kappa", "temperature",
        "use_factored_annotator", "derive_thresholds_from_annotator", "alpha_confusion",
    },
    "tensor": {
        "D",
        "factor_decay",
        "sigma_annotator",
        "sigma_measurement",
        "kappa",
        "temperature",
        # Misspecification flags and loading distribution for tensor-only data generation.
        "use_log_scores",
        "use_logistic_link",
        "use_normal_loadings",
    },
}

# All Stan-data field names that are type-specific (used for validation: only required set may be non-None).
STAN_DATA_FIELDS: frozenset = frozenset({
    "D", "M", "S", "sigma_annotator", "sigma_measurement", "kappa", "temperature",
    "use_factored_annotator", "derive_thresholds_from_annotator", "d_annotator", "factor_decay",
    # Tensor-only misspecification flags / loading distribution.
    "use_log_scores", "use_logistic_link", "use_normal_loadings",
    # Dawid-Skene confusion prior.
    "alpha_confusion",
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
    - Core: K_train, K_test, K_val, I, J, C, enable_pairwise_rankings, pairwise_cap_per_item.
    - All other Stan-related fields are explicit attributes, default None. For a given
      stan_type, exactly the set in STAN_TYPE_REQUIRED[stan_type] must be set (non-None);
      all other Stan-data fields must be None (enforced by check_config_for_stan_type).
    - to_stan_data() validates and returns a single Stan data dict (core + type-specific
      fields from this config), used for data generation and for saving stan_data.json.
    """

    # --- Core (always required) ---
    K_train: int
    K_test: int
    K_val: int
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
    alpha_confusion: Optional[float] = None  # Dawid-Skene confusion prior concentration

    # Misspecification flags (tensor-only). These are part of Stan data for the tensor
    # data-generation model and are required (0/1) for stan_type="tensor".
    use_log_scores: Optional[int] = None       # 0/1: apply log() to raw CP scores before binning
    use_logistic_link: Optional[int] = None    # 0/1: use inv_logit instead of Phi for binning
    use_normal_loadings: Optional[int] = None  # 0/1: use N(0,1) instead of Exp(1) loadings

    # Observation protocol
    observation_protocol: str = "tie_breaking"  # "tie_breaking", "mcar", "extended_rankings"
    mcar_missing_rate: float = 0.5  # Missing rate for MCAR protocol (was mar_missing_rate)
    pairwise_observation_rate: float = 1.0  # For tie_breaking: fraction of missing pairwise rankings to observe (0.0-1.0)

    seed: Optional[int] = None

    stan_type: str = "factored-dot-product"

    def __post_init__(self) -> None:
        """
        Backwards-compatible defaults for tensor-only misspecification flags.

        Older `configs.json` files produced before these fields were added will not
        contain them; when such a config is reconstructed with stan_type="tensor",
        we treat the missing flags as 0 (off) so that validation still passes and
        Stan data is well-defined.
        """
        if self.stan_type == "tensor":
            if self.use_log_scores is None:
                self.use_log_scores = 0
            if self.use_logistic_link is None:
                self.use_logistic_link = 0
            if self.use_normal_loadings is None:
                self.use_normal_loadings = 0

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
            "K_val": self.K_val,
            "I": self.I,
            "J": self.J,
            "C": self.C,
            "enable_pairwise_rankings": 1 if self.enable_pairwise_rankings else 0,
            "pairwise_cap_per_item": self.pairwise_cap_per_item,
            "num_annotate_annotator": 4
        }

        required = STAN_TYPE_REQUIRED[self.stan_type]
        for key in sorted(required):
            val = getattr(self, key)
            if key in (
                "use_factored_annotator",
                "derive_thresholds_from_annotator",
                "use_log_scores",
                "use_logistic_link",
                "use_normal_loadings",
            ):
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


@dataclass
class AnnotatorSplitConfig:
    """
    Configuration for annotator-split data generation.

    Unlike DataGenConfig (which splits items into train/val/test instances),
    this config keeps a single shared item set and splits annotators:
      - J_train annotators (IDs 1..J_train) rate all K items -> "train" instance
      - J_val annotators (IDs J_train+1..J_train+J_val) rate all K items -> "val" instance
      - J_test annotators (IDs J_train+J_val+1..J) rate all K items -> "test" instance

    Every annotator in each group rates every item (no num_annotate_annotator sampling).
    Pairwise rankings are always disabled for this config.
    """

    # --- Core dimensions ---
    K: int           # Number of shared items (same across all instances)
    J_train: int     # Number of training annotators
    J_val: int       # Number of validation annotators
    J_test: int      # Number of test annotators
    I: int           # Number of attributes
    C: int           # Number of rating categories

    # --- Stan-data fields (same generative model as DataGenConfig) ---
    D: Optional[int] = None
    d_annotator: Optional[int] = None
    sigma_annotator: Optional[float] = None
    sigma_measurement: Optional[float] = None
    kappa: Optional[float] = None
    temperature: Optional[float] = None
    use_factored_annotator: Optional[int] = None
    derive_thresholds_from_annotator: Optional[int] = None

    # Observation protocol
    observation_protocol: str = "mcar"     # "mcar" is the only supported protocol
    mcar_missing_rate: float = 0.5
    enable_pairwise_rankings: bool = False  # always disabled for annotator-split
    pairwise_cap_per_item: int = 10

    seed: Optional[int] = None
    stan_type: str = "factored-dot-product"

    @property
    def J(self) -> int:
        """Total number of annotators across all splits."""
        return self.J_train + self.J_val + self.J_test

    def to_stan_data(self) -> Dict[str, Any]:
        """Build Stan data dict for the annotator-split model."""
        for field in ("D", "d_annotator", "sigma_annotator", "sigma_measurement",
                      "kappa", "temperature", "use_factored_annotator",
                      "derive_thresholds_from_annotator"):
            if getattr(self, field) is None:
                raise ValueError(f"AnnotatorSplitConfig: {field!r} must be set")

        return {
            "K": self.K,
            "J_train": self.J_train,
            "J_val": self.J_val,
            "J_test": self.J_test,
            "J": self.J,
            "I": self.I,
            "C": self.C,
            "D": self.D,
            "d_annotator": self.d_annotator,
            "sigma_annotator": self.sigma_annotator,
            "sigma_measurement": self.sigma_measurement,
            "kappa": self.kappa,
            "temperature": self.temperature,
            "use_factored_annotator": int(self.use_factored_annotator),
            "derive_thresholds_from_annotator": int(self.derive_thresholds_from_annotator),
        }
