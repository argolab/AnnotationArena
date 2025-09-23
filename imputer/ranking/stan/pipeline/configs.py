from dataclasses import dataclass
from typing import Optional


@dataclass
class DataGenConfig:
    K: int
    I: int
    J: int
    D: int
    C: int

    enable_third_annotator: bool = True
    enable_pairwise: bool = True
    pairwise_cap_per_item: int = 10

    sigma_annotator: float = 0.3
    sigma_measurement: float = 0.1
    alpha_dirichlet: float = 2.0
    temperature: float = 0.5

    train_fraction: float = 0.8
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


