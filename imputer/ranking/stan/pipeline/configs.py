from dataclasses import dataclass
from typing import Optional


@dataclass
class DataGenConfig:
    K_train: int
    K_test: int
    I: int
    J: int
    D: int
    C: int

    # Annotator embedding dimension (for low-rank factorization)
    # None means use D (full rank)
    d_annotator: Optional[int] = None

    # Annotator model selection
    # True = new factored model: V_ij = v_i + u_j * M_i (allows covariance learning)
    # False = old spherical model: V_ij ~ N(v_i, sigma^2) independently
    use_factored_annotator: bool = True

    # Rating threshold derivation from annotator embedding
    # True = thresholds derived from u_j (consistent annotator style, fewer d.f.)
    # False = independent Dirichlet samples for each (i,j) pair
    # Only meaningful when use_factored_annotator=True
    derive_thresholds_from_annotator: bool = False

    enable_third_annotator: bool = True
    enable_pairwise_rankings: bool = True
    pairwise_cap_per_item: int = 10

    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5

    # Axis invariance controls for easy-data sanity experiments
    # When a flag is True, ratings should NOT depend on that axis:
    # - hold_I_constant=True  -> no dependence on criteria/attributes i
    # - hold_J_constant=True  -> no dependence on annotator j
    # - hold_K_constant=True  -> no dependence on item k
    #
    # The Stan generator implements this by tying embeddings / rating
    # probabilities / item embeddings across the corresponding indices.
    hold_I_constant: bool = False
    hold_J_constant: bool = False
    hold_K_constant: bool = False

    # CP tensor decomposition parameter
    # T_d = factor_decay^(d-1), controls how fast components decay
    # Only used with tensor_data_generation.stan
    factor_decay: Optional[float] = None

    # Observation protocol
    observation_protocol: str = "tie_breaking"  # "tie_breaking", "mcar", "extended_rankings"
    mcar_missing_rate: float = 0.5  # Missing rate for MCAR protocol (was mar_missing_rate)
    pairwise_observation_rate: float = 1.0  # For tie_breaking: fraction of missing pairwise rankings to observe (0.0-1.0)

    seed: Optional[int] = None


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
    alpha_dirichlet: float = 2.0
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


