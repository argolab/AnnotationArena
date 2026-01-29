## Easy-data axis invariance sanity ladder

This experiment suite is designed to debug the Marformer imputer by starting
from extremely simple synthetic data and gradually increasing difficulty,
while keeping the **Stan domain model** and **data format** unchanged.

The ladder is implemented via three axis invariance flags in the Stan
data generator (`models/iclr_data_generation.stan`):

- `hold_I_constant` (criteria / attributes)
- `hold_J_constant` (annotators)
- `hold_K_constant` (items)

When a flag is `1`, ratings are constructed so that they **do not depend on
that axis**. This is implemented in the generator by tying:

- `mean_preferences[i, :]` (criteria embeddings),
- `annotator_preferences[(i-1)*J + j, :]` (annotator embeddings),
- `rating_probs[(i-1)*J + j]` / thresholds,
- and the item embeddings `e_k`,

across the corresponding indices. The observation protocol, binning, and
pairwise construction logic are left unchanged.

### Axis modes

The following modes are available via the three flags:

- **constant**: `hold_I=1, hold_J=1, hold_K=1`
  - Ratings are invariant to I, J, and K (single global structure).
- **I-only**: `hold_I=0, hold_J=1, hold_K=1`
  - Ratings depend only on attribute `i`, not on annotator or item.
- **J-only**: `hold_I=1, hold_J=0, hold_K=1`
  - Ratings depend only on annotator `j`, not on attribute or item.
- **K-only**: `hold_I=1, hold_J=1, hold_K=0`
  - Ratings depend only on item `k`, not on attribute or annotator.
- **IJ**: `hold_I=0, hold_J=0, hold_K=1`
  - Ratings depend on `(i, j)`, but are constant across items.
- **IK**: `hold_I=0, hold_J=1, hold_K=0`
  - Ratings depend on `(i, k)`, but are constant across annotators.
- **JK**: `hold_I=1, hold_J=0, hold_K=0`
  - Ratings depend on `(j, k)`, but are constant across criteria.
- **IJK**: `hold_I=0, hold_J=0, hold_K=0`
  - Full dependence on all three axes (the original complex setting).

### How to run the ladder

Use the convenience script:

```bash
bash scripts/easy_data/easy_all_modes.sh
```

This script loops over all of the above modes and, for each one:

1. Calls
   `python stan/scripts/generate_data.py`
   with the appropriate `--hold-I-constant`, `--hold-J-constant`,
   and `--hold-K-constant` flags to generate a Stan data bundle.
2. Runs the Marformer imputer via
   `python imputer/run_imputer.py`
   on the generated data.
3. Runs Stan inference (4-chain and 1-chain) using
   `stan/scripts/run_inference.py`.
4. Evaluates Stan predictions with
   `stan/scripts/evaluate_predictions.py`.
5. Generates comparison plots with
   `python utils/visualize.py`.

Output locations follow the existing convention:

- Data: `OUTPUT/generated_data/easy_axis_*`
- Imputer runs: `OUTPUT/IMPUTER/easy_axis_*`
- Stan runs: `OUTPUT/domain_model/runs/easy_axis_*`
- Stan eval: `OUTPUT/domain_model/eval/easy_axis_*`

### What to expect / how to interpret failures

- **constant mode**
  - **Expectation**: near-perfect reconstruction.
    - Imputer train / test rating accuracy ≈ 1.0,
      masked and observed losses → 0.
    - Stan predictive metrics should also be essentially perfect.
  - **If this fails**:
    - Look for bugs in masking, loss definition, or data loading
      (this is the most trivial setting).

- **Single-axis modes (I-only, J-only, K-only)**
  - **Expectation**: very high accuracy and fast convergence.
    - I-only: embeddings should effectively ignore `j`/`k`; learned
      structure should primarily track attributes.
    - J-only: per-annotator biases should be captured easily.
    - K-only: model is essentially memorizing per-item ratings.
  - **If these fail but constant works**:
    - The model may be struggling with simple structured variation
      under the current masking rate or architecture; inspect training
      curves and attention patterns.

- **Two-axis modes (IJ, IK, JK)**
  - **Expectation**: still near-perfect with enough epochs, but
    slightly harder than single-axis modes due to interactions.
  - **If these fail but single-axis succeed**:
    - Suspect issues with modeling interactions (capacity, masking,
      or how inputs are encoded).

- **Full IJK**
  - **Expectation**: behavior similar to your current complex setup.
  - **If all easy modes succeed but IJK fails**:
    - The failure is likely specific to the full embedding + noise +
      pairwise complexity, not to the basic training machinery.
    - Compare imputer performance directly against the Stan metrics
      for IJK runs to see whether the issue is capacity, optimization,
      or mismatch between the neural objective and the domain model.

