"""
Test suite for per-stan_type config validation.

For each stan_type we test:
- Valid config (exactly the required fields set) -> check_config_for_stan_type passes, to_stan_data() succeeds.
- Missing a required parameter -> ValueError.
- Extra parameter set (a type-specific field not in the required set) -> ValueError.
"""

import pytest

from stan.pipeline.configs import (
    DataGenConfig,
    STAN_TYPE_REQUIRED,
    STAN_DATA_FIELDS,
    check_config_for_stan_type,
)


# Shared core for all configs (minimal valid dimensions).
CORE = {
    "K_train": 5,
    "K_test": 3,
    "I": 2,
    "J": 4,
    "C": 3,
}


def _make_config(stan_type: str, **type_specific) -> DataGenConfig:
    """Build a DataGenConfig with core + stan_type and only the given type_specific kwargs."""
    return DataGenConfig(stan_type=stan_type, **CORE, **type_specific)


# -----------------------------------------------------------------------------
# Valid type-specific fields per stan_type (all required, no extra).
# -----------------------------------------------------------------------------

VALID_DISCRETE = {
    "M": 6,
    "S": 3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5,
}

VALID_NORMAL_NOISE_DOT_PRODUCT = {
    "D": 8,
    "d_annotator": 8,
    "sigma_annotator": 0.3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5,
    "use_factored_annotator": 0,
    "derive_thresholds_from_annotator": 0,
}

VALID_FACTORED_DOT_PRODUCT = {
    "D": 8,
    "d_annotator": 8,
    "sigma_annotator": 0.3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5,
    "use_factored_annotator": 1,
    "derive_thresholds_from_annotator": 0,
}

VALID_TENSOR = {
    "D": 8,
    "factor_decay": 0.9,
    "sigma_annotator": 0.3,
    "sigma_measurement": 0.1,
    "alpha_dirichlet": 2.0,
    "temperature": 0.5,
    "use_log_scores": 0,
    "use_logistic_link": 0,
    "use_normal_loadings": 0,
}

VALID_BY_TYPE = {
    "discrete": VALID_DISCRETE,
    "normal-noise-dot-product": VALID_NORMAL_NOISE_DOT_PRODUCT,
    "factored-dot-product": VALID_FACTORED_DOT_PRODUCT,
    "tensor": VALID_TENSOR,
}


# -----------------------------------------------------------------------------
# Tests: valid config for each type
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("stan_type", list(STAN_TYPE_REQUIRED))
def test_valid_config_check_passes(stan_type: str):
    """For each stan_type, a config with exactly the required fields set passes check_config_for_stan_type."""
    kwargs = VALID_BY_TYPE[stan_type].copy()
    config = _make_config(stan_type, **kwargs)
    check_config_for_stan_type(config)


@pytest.mark.parametrize("stan_type", list(STAN_TYPE_REQUIRED))
def test_valid_config_to_stan_data_succeeds(stan_type: str):
    """For each stan_type, to_stan_data() returns a dict with core + type-specific keys."""
    kwargs = VALID_BY_TYPE[stan_type].copy()
    config = _make_config(stan_type, **kwargs)
    out = config.to_stan_data()
    assert "K_train" in out and out["K_train"] == CORE["K_train"]
    assert "I" in out and out["I"] == CORE["I"]
    required = STAN_TYPE_REQUIRED[stan_type]
    for key in required:
        assert key in out, f"Missing required key {key!r} in to_stan_data() output"
        assert out[key] is not None


# -----------------------------------------------------------------------------
# Tests: missing a required parameter -> ValueError
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("stan_type", list(STAN_TYPE_REQUIRED))
def test_missing_one_required_raises(stan_type: str):
    """Omitting any single required field causes check_config_for_stan_type to raise ValueError."""
    required = STAN_TYPE_REQUIRED[stan_type]
    for omit_key in required:
        kwargs = {k: v for k, v in VALID_BY_TYPE[stan_type].items() if k != omit_key}
        config = _make_config(stan_type, **kwargs)
        with pytest.raises(ValueError) as exc_info:
            check_config_for_stan_type(config)
        assert omit_key in str(exc_info.value) or "must set" in str(exc_info.value).lower()


@pytest.mark.parametrize("stan_type", list(STAN_TYPE_REQUIRED))
def test_missing_one_required_to_stan_data_raises(stan_type: str):
    """Omitting any single required field causes to_stan_data() to raise ValueError."""
    required = STAN_TYPE_REQUIRED[stan_type]
    omit_key = next(iter(required))
    kwargs = {k: v for k, v in VALID_BY_TYPE[stan_type].items() if k != omit_key}
    config = _make_config(stan_type, **kwargs)
    with pytest.raises(ValueError):
        config.to_stan_data()


# -----------------------------------------------------------------------------
# Tests: extra parameter set -> ValueError
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("stan_type", list(STAN_TYPE_REQUIRED))
def test_extra_parameter_raises(stan_type: str):
    """Setting any type-specific field that is not in the required set raises ValueError."""
    required = STAN_TYPE_REQUIRED[stan_type]
    extra_candidates = STAN_DATA_FIELDS - required
    if not extra_candidates:
        pytest.skip(f"No extra fields for {stan_type} (required set equals all Stan-data fields)")
    kwargs = dict(VALID_BY_TYPE[stan_type])
    extra_key = next(iter(extra_candidates))
    # Set an extra field to a non-None value (choose a sensible default).
    if extra_key in ("D", "d_annotator", "M", "S"):
        kwargs[extra_key] = 1
    elif extra_key == "factor_decay":
        kwargs[extra_key] = 0.8
    elif extra_key in (
        "use_factored_annotator",
        "derive_thresholds_from_annotator",
        "use_log_scores",
        "use_logistic_link",
        "use_normal_loadings",
    ):
        kwargs[extra_key] = 0
    else:
        kwargs[extra_key] = 0.5
    config = _make_config(stan_type, **kwargs)
    with pytest.raises(ValueError) as exc_info:
        check_config_for_stan_type(config)
    assert "must not set" in str(exc_info.value).lower() or extra_key in str(exc_info.value)


@pytest.mark.parametrize("stan_type", list(STAN_TYPE_REQUIRED))
def test_extra_parameter_to_stan_data_raises(stan_type: str):
    """Setting an extra type-specific field causes to_stan_data() to raise ValueError."""
    required = STAN_TYPE_REQUIRED[stan_type]
    extra_candidates = STAN_DATA_FIELDS - required
    if not extra_candidates:
        pytest.skip(f"No extra fields for {stan_type}")
    kwargs = dict(VALID_BY_TYPE[stan_type])
    extra_key = next(iter(extra_candidates))
    if extra_key in ("D", "d_annotator", "M", "S"):
        kwargs[extra_key] = 1
    elif extra_key == "factor_decay":
        kwargs[extra_key] = 0.8
    elif extra_key in (
        "use_factored_annotator",
        "derive_thresholds_from_annotator",
        "use_log_scores",
        "use_logistic_link",
        "use_normal_loadings",
    ):
        kwargs[extra_key] = 0
    else:
        kwargs[extra_key] = 0.5
    config = _make_config(stan_type, **kwargs)
    with pytest.raises(ValueError):
        config.to_stan_data()


# -----------------------------------------------------------------------------
# Unknown stan_type
# -----------------------------------------------------------------------------

def test_unknown_stan_type_raises():
    """An unknown stan_type raises ValueError."""
    config = _make_config("discrete", **VALID_DISCRETE)
    config.stan_type = "unknown-type"
    with pytest.raises(ValueError) as exc_info:
        check_config_for_stan_type(config)
    assert "Unknown stan_type" in str(exc_info.value)
