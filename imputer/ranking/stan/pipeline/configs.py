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

    enable_third_annotator: bool = True
    enable_pairwise_rankings: bool = True
    pairwise_cap_per_item: int = 10

    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5

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


