# Imputer Ablation Study

Temporary ablation flags for leave-one-out feature ablation on the Imputer (Marformer) pipeline.

## Ablation Flags

All flags are prefixed with `--ablation-no-xxx` and default to `False`. When set, they remove the corresponding feature from the training pipeline:

| Flag | Effect |
|------|--------|
| `--ablation-no-dropout` | Set dropout, item_embedding_dropout, annotator_dropout to 0 |
| `--ablation-no-mask-augmentation` | Set mask_augmentations=1 (no fresh random masks per epoch) |
| `--ablation-no-random-masking` | Set masking_rate=1.0 (mask all observed tokens) |
| `--ablation-no-cosine-schedule` | Disable cosine LR schedule (constant LR) |
| `--ablation-no-transductive` | Disable transductive learning (train partition only) |
| `--ablation-no-pointer-mechanism` | Disable pointer/edge-indicator attention mechanism |
| `--ablation-no-parameter-normalization` | Disable parameter normalization in transformer blocks |
| `--ablation-no-human-supervision` | Set human_observed_rate=0 (LLM-only supervision) |
| `--ablation-no-weight-decay` | Set weight_decay=0 |

## Running the Ablation Sweep

```bash
# From repo root
bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh

# Smoke run: 5 epochs per ablation (quick validation)
SMOKE_RUN=1 bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh

# With dist bundle (soft labels)
BUNDLE=dist bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh

# Dry run (echo commands only)
DRY_RUN=1 bash scripts/real_data/run_real_data_inputer_ablation_sweeps.sh
```

The sweep runs 10 configurations: baseline (BASE) plus one ablated variant for each feature. With `SMOKE_RUN=1`, each run trains for only 5 epochs and outputs go under run names like `llm_rubric_marformer_ablation_smoke_BASE` (separate from full runs). After training, it generates:

- **Summary table**: `OUTPUT/IMPUTER/llm_rubric_imputer_ablation_summary.csv`  
  Columns: ablation_id, test_missing_xent_last{1,5,10}, train_xent_last{1,5,10}, test_missing_acc_last1

- **Plots**: `OUTPUT/IMPUTER/plots/IMPUTER_ABLATION/`  
  - `train_loss_curves.png`  
  - `test_missing_xent_curves.png`  
  - `test_missing_acc_curves.png`  

## Standalone Summarization and Plotting

If you already have ablation run directories, you can run the scripts directly:

```bash
python scripts/real_data/summarize_imputer_ablation.py \
  --output-root OUTPUT/IMPUTER \
  --run-prefix llm_rubric_marformer_ablation

python scripts/real_data/plot_imputer_ablation_curves.py \
  --output-root OUTPUT/IMPUTER \
  --run-prefix llm_rubric_marformer_ablation \
  --plot-dir OUTPUT/IMPUTER/plots/IMPUTER_ABLATION
```

## Temporary Nature

These flags and scripts are for the ablation study only. They can be removed once the study is complete.
