# Stan data: config and per-type fields (data generation)

This document is the **source of truth** for how Stan data is produced from `DataGenConfig`. There is **no separate "extra" dict**: all Stan-related fields are **explicit attributes** on the config (default `None`). For each `stan_type`, **exactly** the set of type-specific fields required for that type must be set on the config; all other type-specific fields must remain `None`. `config.to_stan_data()` validates and returns the single Stan data dict (core + type-specific) used for data generation and for saving `stan_data.json` for inference.

---

## 1. Core Stan data (all types)

These keys are always present in Stan data and come from the config.

| Key | Type | Description |
|-----|------|-------------|
| `K_train` | int | Number of items in training instance |
| `K_test` | int | Number of items in test instance |
| `I` | int | Number of criteria (attributes) |
| `J` | int | Number of annotators |
| `C` | int | Number of rating categories |
| `enable_pairwise_rankings` | 0/1 | Enable pairwise rankings |
| `pairwise_cap_per_item` | int | Max pairwise comparisons per item |

**Implementation:** `DataGenConfig` has these as required (or defaulted) fields. `to_stan_data()` always includes them.

---

## 2. Type-specific fields (all on config, default `None`)

Every Stan-data field that can vary by model is an **explicit attribute** on `DataGenConfig`, each defaulting to `None`:

- `D`, `M`, `S`
- `sigma_annotator`, `sigma_measurement`, `kappa`, `temperature`
- `use_factored_annotator`, `derive_thresholds_from_annotator` (0/1)
- `d_annotator`, `factor_decay`
- Tensor-only misspecification / loading flags (0/1): `use_log_scores`, `use_logistic_link`, `use_normal_loadings`

For a given `stan_type`, **exactly** the set of fields listed for that type in §3 must be set (non-`None`); **no other** type-specific field may be set. Validation is done by `check_config_for_stan_type(config)` and by `config.to_stan_data()`.

**Data generation .stan by type:**

| stan_type | Data generation .stan |
|-----------|------------------------|
| `normal-noise-dot-product` | `iclr_data_generation.stan` |
| `factored-dot-product` | `iclr_data_generation.stan` |
| `discrete` | `discrete_type_data_generation.stan` |
| `tensor` | `tensor_data_generation.stan` |

**Pairing with domain models:** The same set of type-specific fields is shared between data generation and the paired domain model (e.g. `discrete_type_domain_model.stan` for `discrete`, `domain_model.stan` for the two dot-product types, `tensor_domain_model.stan` for `tensor`).

---

## 3. Required fields per stan_type

### 3.1 `discrete`

- **Required (exactly):** `M`, `S`, `sigma_measurement`, `kappa`, `temperature`
- **Meaning:** M item prototypes, S annotator styles; measurement noise, Dirichlet concentration, and temperature for pairwise generation. `sigma_annotator` and `D` do **not** apply to discrete; the discrete Stan data block still accepts `D` and `sigma_annotator` for compatibility (filled with placeholders in `to_stan_data()`).
- **Not exposed:** `sigma_rubric_fuzz` is fixed in the data-generation Stan and inferred by the domain model, not passed as data.

**CLI example:**  
`--stan-type discrete --stan-arg M=6 --stan-arg S=3`  
(Other params like `sigma_measurement`, `kappa`, `temperature` come from CLI defaults or `--stan-arg`.)

### 3.2 `normal-noise-dot-product`

- **Required (exactly):** `D`, `d_annotator`, `sigma_annotator`, `sigma_measurement`, `kappa`, `temperature`, `use_factored_annotator`, `derive_thresholds_from_annotator`
- **Meaning:** Embedding dimension D; annotator dimension `d_annotator`; annotator/measurement noise; Dirichlet (kappa) and temperature; `use_factored_annotator=0`, `derive_thresholds_from_annotator=0` (spherical annotator model).

### 3.3 `factored-dot-product`

- **Required (exactly):** `D`, `d_annotator`, `sigma_annotator`, `sigma_measurement`, `kappa`, `temperature`, `use_factored_annotator`, `derive_thresholds_from_annotator`
- **Meaning:** Same as above with `use_factored_annotator=1`; `derive_thresholds_from_annotator` is 0 or 1 (thresholds from annotator embedding or independent Dirichlet).

### 3.4 `tensor`

- **Required (exactly):**
  - `D`, `factor_decay`, `sigma_annotator`, `sigma_measurement`, `kappa`, `temperature`
  - Tensor-only misspecification / loading flags (all 0/1):  
    `use_log_scores`, `use_logistic_link`, `use_normal_loadings`
- **Meaning:**
  - Core CP tensor hyperparameters: all dimensions match (rank `D`). `d_annotator` is set to `D` automatically in the pipeline (not a separate parameter). `factor_decay` is the decay for factor weights.
  - Misspecification / loading flags (used only by `tensor_data_generation.stan` for robustness experiments):
    - `use_log_scores`: apply `log()` to raw CP scores before binning into Likert categories (misspecifies the linear Gaussian likelihood assumed by the domain model).
    - `use_logistic_link`: use `inv_logit` instead of `Phi` for binning (misspecifies the probit link).
    - `use_normal_loadings`: use `N(0,1)` loadings instead of `Exp(1)` for the CP factors (allows signed cancellation).

Typical CLI usage for tensor data generation:

- Base hyperparameters (from flags or defaults): e.g. `--D 8 --sigma-annotator 0.3 --sigma-measurement 0.1 --kappa 2.0`.
- Tensor-specific fields via `--stan-arg` (overriding any defaults):  
  `--stan-type tensor --stan-arg factor_decay=0.9 --stan-arg use_log_scores=1 --stan-arg use_logistic_link=1 --stan-arg use_normal_loadings=1`.

---

## 4. Pipeline flow (reference)

1. **Data generation (`stan/scripts/generate_data.py`)**
   - Parse CLI and `--stan-arg KEY=VALUE`.
   - Build `DataGenConfig`: set core (K_train, K_test, I, J, C, etc.) and **only** the type-specific fields for the chosen `stan_type` (from args and/or `--stan-arg`). All other type-specific fields remain `None`.
   - Call `generate_data(config, stan_file)`; inside, `stan_data = config.to_stan_data()` (validates and builds the dict).
   - Save `configs.json` (full datagen config, including type-specific fields) and `stan_data.json` for inference.

2. **Inference (`stan/scripts/run_inference.py` + `stan/pipeline/inference.py`)**
   - Load bundle and `configs.json`; reconstruct `DataGenConfig`.
   - Choose domain model via `--stan-type` (or from config). Model-specific Stan data is passed via **`--stan-arg KEY=VALUE`** and merged after `config.to_stan_data()`.
   - **Cross-type inference:** The *data* config is from the bundle (data-generation type), so it does not contain the *inference* model’s type-specific parameters. The **caller** (e.g. `scripts/cross_stan_type_experiment.py`) is the single source of truth: it must pass all parameters required by the chosen domain model via `--stan-arg`. `run_inference.py` does not inject any defaults.
   - `prepare_stan_data_for_inference(bundle, config, ..., stan_arg=...)` builds observed-data dict, merges **`config.to_stan_data()`**, then overlays `stan_arg`.

---

## 5. Implementation summary

- **`stan/pipeline/configs.py`:** `STAN_TYPE_REQUIRED` defines the exact set of type-specific field names per `stan_type`. `check_config_for_stan_type(config)` enforces that for `config.stan_type` exactly those fields are non-`None` and all others are `None`. `DataGenConfig.to_stan_data()` calls that check and returns core + type-specific fields as a single dict.
- **No `stan_data_extra`:** All values are on the config; nothing is merged from a separate extra dict.
- **CLI:** Sets only the required fields for the chosen `stan_type` on the config; other type-specific fields are left `None` by default.
