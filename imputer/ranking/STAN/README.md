# STAN

Stan pipeline for the tensor-prototype "Compositional Projection Model": generates
synthetic annotation data, then fits a correctly-specified ("oracle") and three
structurally misspecified inference models against it. Used by
`../scripts/DOMAIN3-SIGMOID/` and `../scripts/LLMRubric/CPMSigmoid/` to produce
Stan baselines compared against Marformer. 

## stan_models/

Five `.stan` files sharing one data schema:

| File | Role |
|---|---|
| `tensor_generation_sigmoid.stan` | Data generator. |
| `tensor_model_sigmoid.stan` | Oracle — correctly-specified inference model. |
| `tensor_shared_annotator_binning_raw_z_sigmoid.stan` | Misspec "Bin" — shares rating thresholds across annotators. |
| `tensor_shared_annotator_projection_sigmoid.stan` | Misspec "Proj" — shares the annotator→prototype projection. |
| `tensor_shared_annotator_projection_binning_raw_z_sigmoid.stan` | Misspec "Both" — combines Bin and Proj. |

## stan_code/pipeline/

Core library, imported by the CLI scripts below.

| File | Contents |
|---|---|
| `configs.py` | `DataGenConfig` / `AnnotatorSplitConfig` dataclasses and Stan data field validation. |
| `data_gen.py` | Runs the generator Stan model, converts output to a `GroundTruthBundle`. |
| `inference.py` | `InferenceConfig`, model compilation, ground-truth MCMC init helpers. |
| `bundle.py` | `GroundTruthBundle` — the shared schema for ratings, pairwise comparisons, ground truth. |
| `io.py` | Run-directory creation, JSON/bundle save helpers. |
| `predictives.py` | Extracts posterior predictive samples, computes log-loss against held-out ratings. |

## stan_code/scripts/

**Generation**
- `generate_data.py` — item-split generator CLI.
- `generate_data_annotator.py` — annotator-split generator (needs `--stan-file`; no default generator model exists for this split yet).
- `generate_data_item_annotator.py` — joint item+annotator split generator.
- `domain3subsplit.py` — builds the DOMAIN3 item/annotator expansion splits from one generated bundle; called by `generate_data.sh`.

**Subsetting** (shrink an existing generated directory to a smaller train size)
- `subset_item_split.py`, `subset_annotator_split.py`, `subset_item_annotator_split.py`

**Inference & evaluation**
- `run_inference.py` — MCMC fitting CLI; `--model-variant {oracle,misspec-bin,misspec-proj,misspec-both}` selects a model by name, or pass `--stan-file` directly.
- `evaluate_predictions.py` — scores posterior predictions against ground truth.
- `smooth_predictions.py` — post-hoc temperature scaling of rating-probability outputs.

## Example

```bash
python STAN/stan_code/scripts/generate_data.py \
    --output-dir runs/data_gen_test --K-train 10 --K-test 10 --K-val 10 --I 3 --J 9

python STAN/stan_code/scripts/run_inference.py \
    --data-bundle runs/data_gen_test/data_bundle.json \
    --configs     runs/data_gen_test/configs.json \
    --output-dir  runs/inference_test --model-variant oracle
```

The generic runner that `../scripts/DOMAIN3-SIGMOID/run_stan_group.sh` delegates
to (env setup, `.stan` file staging, the fit+eval loop) lives in
`../scripts/DOMAIN3/`.
