"""Structured baselines: unigram (ij), IJK NB, structured NB, optional log-linear."""

from .dataset_adapter import (
    LocalExample,
    build_eval_examples,
    build_test_examples,
    build_train_observed_examples,
    bundle_dims,
    load_bundle_dict,
    transductive_observed_cells,
)
from .factor_routing import FactorKind, RoutedSources, route_source, route_sources
from .log_linear_structured import StructuredLogLinear
from .runner import (
    FittedBaselines,
    calibration_probs_labels,
    evaluate_split,
    fit_baselines,
    load_and_fit,
)
from .naive_bayes_ijk import NaiveBayesIJK
from .naive_bayes_structured import StructuredNaiveBayes
from .unigram_pooled import PooledUnigramIJ

__all__ = [
    "LocalExample",
    "FittedBaselines",
    "fit_baselines",
    "load_and_fit",
    "evaluate_split",
    "calibration_probs_labels",
    "PooledUnigramIJ",
    "NaiveBayesIJK",
    "StructuredNaiveBayes",
    "StructuredLogLinear",
    "FactorKind",
    "RoutedSources",
    "route_source",
    "route_sources",
    "bundle_dims",
    "build_test_examples",
    "build_eval_examples",
    "build_train_observed_examples",
    "load_bundle_dict",
    "transductive_observed_cells",
]
