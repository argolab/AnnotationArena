# Experimental and Script Pipeline

All paths are relative to `imputer/ranking/` (the working directory for all commands).

---

## Directory Structure

```
scripts/
├── stan/                         # Synthetic data experiments (lowercase)
│   ├── generate_data_itemtest.sh
│   ├── generate_data_annotatortest.sh
│   ├── stan_data_command_marformer.sh       # Normal ItemTest generation
│   ├── stan_data_command_marformer_hard.sh  # Factor ItemTest generation
│   ├── stan_data_command_marformer_annotator.sh  # AnnotatorTest generation
│   ├── data_split.sh
│   ├── run_inference.sh                     # STAN baseline inference (Normal/Factor)
│   ├── MARFORMER/                           # Non-transductive training
│   │   ├── Factor_250_20_9_AnnotatorTest/
│   │   │   └── <split>/run_train.sh
│   │   ├── Factor_650_20_9_ItemTest/
│   │   ├── Normal_250_20_9_AnnotatorTest/
│   │   └── Normal_650_20_9_ItemTest/
│   ├── MARFORMER_HARD/                      # Transductive + VTM, ItemTest
│   │   ├── Factor_650_20_9_ItemTest/
│   │   │   └── <split>/run_train.sh
│   │   └── Normal_650_20_9_ItemTest/
│   ├── MARFORMER_ANNOT_DROP/               # Transductive + VTM, AnnotatorTest
│   │   ├── Factor_250_20_9_AnnotatorTest/
│   │   │   └── <split>/run_train.sh  (+ run_train_transductive_mask*.sh)
│   │   └── Normal_250_20_9_AnnotatorTest/
│   │       └── <split>/run_train.sh  (+ VALTEST_MASK/ subfolder)
│   ├── HP_SEARCH/                           # Hyperparameter search scripts
│   └── SPARSE/                              # SPARSE ItemTest experiments
│       ├── Factor_225_25_9_ItemTest_Cluster/     # Factor SPARSE Marformer (SLURM)
│       │   ├── run_small_sizes.sh           # sizes 10,20,30,50; MASKING_RATE=0.50
│       │   ├── run_size100.sh
│       │   └── run_size175.sh
│       ├── Factor_225_25_9_ItemTest_Local/       # Factor SPARSE local variants
│       ├── Factor_225_25_9_ItemTest_Size_175/    # Factor size-175 specific
│       ├── Normal_225_25_9_ItemTest_Size_175/    # Normal SPARSE size-175
│       ├── DawidSkene_225_25_9_ItemTest/         # DS Marformer training (SLURM + local)
│       │   ├── generate_data.sh             # generates K_train=200 then subsets
│       │   ├── run_small_sizes.sh           # sizes 10,30,50,75; MASKING_RATE=0.15
│       │   ├── run_size100.sh
│       │   ├── run_size175.sh
│       │   ├── run_size200.sh
│       │   ├── run_size10_local.sh          # local CPU test
│       │   └── run_eval_test.sh             # local eval on test split (all sizes)
│       └── DawidSkene_225_25_9_ItemTest_Stan/    # DS Stan HMC inference (SLURM)
│           ├── run_size10.sh
│           ├── run_size30.sh
│           ├── run_size50.sh
│           ├── run_size75.sh
│           ├── run_size100.sh
│           ├── run_size150.sh
│           ├── run_size175.sh
│           └── run_size200.sh
│
├── LLM_RUBRIC/                   # LLMRubric real data
│   ├── MARFORMER/TRAIN/
│   │   ├── run_train.sh          # Standard (non-transductive or transductive)
│   │   └── run_train_granular.sh # More granular logging splits
│   └── BASELINE_MLP/
│
├── SUMMEVAL/                     # SummEval real data
│   ├── MARFORMER/TRAIN/
│   │   ├── run_train.sh          (K_train = 100 default)
│   │   ├── run_train_a.sh
│   │   ├── run_train_b.sh        (K_train = 500)
│   │   ├── run_train_b_750.sh
│   │   ├── run_train_b_1000.sh
│   │   └── run_train_b_1280.sh
│   └── BASELINE_MLP/
│
├── convert-llm-rubric/           # Raw → bundle conversion
└── convert-summeval/
```

---

## Step 1: Data Preparation

### Synthetic Data (STAN)

```bash
# Generate Factor + Normal ItemTest bundles for all K_test splits
bash scripts/stan/generate_data_itemtest.sh

# Generate Factor + Normal AnnotatorTest bundles for all J_test splits
bash scripts/stan/generate_data_annotatortest.sh

# Generate Dawid-Skene SPARSE ItemTest (K_train=200 then subsets to 175/150/100/75/50/30/10)
bash scripts/stan/SPARSE/DawidSkene_225_25_9_ItemTest/generate_data.sh
```

Outputs land in `DATA/STAN/<family>/<split>/data_bundle.json` or
`DATA/STAN/SPARSE/<family>/<split>/data_bundle.json` for SPARSE families.

### Real Data

```bash
# Convert LLMRubric from raw format to bundle
python scripts/convert-llm-rubric/convert.py ...

# Convert SummEval
python scripts/convert-summeval/ ...
```

Outputs land in `DATA/LLM_RUBRIC/<split>/data_bundle.json` and `DATA/SUMMEVAL/<split>/`.

---

## Step 2: Training

### Entry Point

```bash
python -u -m imputer.entity_mf.train \
    --data-dir   DATA/STAN/<family>/<split> \
    --output-root RESULTS/<family> \
    --run-name   <run_name> \
    [hyperparams and flags]
```

### Naming Convention for Script Families

| Script family | Mode | Data | Key flags |
|---|---|---|---|
| `MARFORMER` | Non-transductive (default) | STAN, LLMRubric, SummEval | `--use-pointer`, no transductive |
| `MARFORMER_HARD` | Transductive + VTM | STAN ItemTest | `--transductive-learning --transductive-valtest-mask --use-graph-mask` |
| `MARFORMER_ANNOT_DROP` | Transductive + VTM | STAN AnnotatorTest | same + annotator dropout variants |
| `MARFORMER_HARD_MASK` | Transductive + VTM | LLMRubric, SummEval | same as MARFORMER_HARD |
| `SPARSE/*_CLUSTER_NOITEMDEV_TRANS` | Transductive (no VTM) | SPARSE ItemTest (Factor, Normal, DawidSkene) | `--transductive-learning` only; Factor/Normal: rate=0.50, DawidSkene: rate=0.15 |

### SLURM (Cluster)

All cluster scripts use the same pattern and are submitted via `sbatch_adapt`:

```bash
cd /path/to/imputer/ranking
PARTITION=gpu-a100 GPUS=1 TIME=48:00:00 CPUS_PER_TASK=16 MEM_PER_CPU=18G \
  /home/xwang397/bin/sbatch_adapt scripts/STAN/MARFORMER/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_3/run_train.sh
```

Header variables per script: `#SBATCH --account=a100acct`, `#SBATCH --partition=gpu-a100`,
environment: `llm_rubric_env`, base dir: `/export/fs06/psingh54/`.

### Local Testing

Local scripts (`run_train_local*.sh`) use `DEVICE=cpu` and reduced epochs for quick iteration.
These are in the same split directories alongside cluster scripts.

---

## Step 3: Key Hyperparameters per Experiment Family

### MARFORMER (non-transductive baseline)

```bash
EMBEDDING_DIM=80
NUM_LAYERS=8
ATTENTION_HEADS=4
D_FF=128
EPOCHS=200
LR=2e-4
MASKING_RATE=0.15
MASK_AUGMENTATIONS=5
ITEM_DROPOUT_RATE=0.7
USE_POINTER=true
USE_GRAPH_MASK=false
```

### MARFORMER_HARD / MARFORMER_ANNOT_DROP (transductive + VTM)

```bash
EPOCHS=300
LR=2e-4
MASKING_RATE=0.5              # higher masking since pool is only val+test observed
ITEM_DROPOUT_RATE=0.3         # or 0.0 for AnnotatorTest
ANNOTATOR_DROPOUT_RATE=0.0    # or 0.3/0.7 in dropout ablation variants
USE_GRAPH_MASK=true           # hard graph mask enabled
USE_TRANSDUCTIVE_VALTEST_MASK=true
--transductive-learning
--transductive-valtest-mask
```

---

## Step 4: Output Structure

```
RESULTS/<family>/STAN/<run_name>/
    config.json        — full hyperparameter snapshot
    best_model.pt      — best checkpoint by val loss
    metrics.json       — final eval metrics (accuracy, CE, per-split)
    training_history.json
    lightning_logs/
```

---

## Step 5: Evaluation and Inference

```bash
# Run STAN inference baseline on a bundle (Normal/Factor)
bash scripts/stan/run_inference.sh

# Evaluate STAN predictions vs ground truth
python STAN/stan_code/scripts/evaluate_predictions.py --bundle DATA/STAN/...

# Dawid-Skene STAN HMC inference + eval (one script per size)
bash scripts/stan/SPARSE/DawidSkene_225_25_9_ItemTest_Stan/run_size200.sh

# Evaluate Marformer test-split predictions (local, all DawidSkene sizes)
bash scripts/stan/SPARSE/DawidSkene_225_25_9_ItemTest/run_eval_test.sh
```

The `run_eval_test.sh` script iterates over all 8 sizes, calls
`python -m imputer.entity_mf.test --run-dir ... --checkpoint best --device cpu`,
and saves results to `RESULTS/MARFORMER/STAN/SPARSE/<RUN_NAME>/TEST_RESULTS/best.json`.

---

## Step 6: Data Integrity Check

```bash
# Run from imputer/ranking (not tracked in git)
python test_data/check_bundle.py --data-dir DATA/STAN/Factor_650_20_9_ItemTest/Factor_650_20_9_ItemTest_100
python test_data/check_bundle.py --data-dir DATA/STAN/Factor_250_20_9_AnnotatorTest/Factor_250_20_9_AnnotatorTest_3
```

Checks: duplicates, coverage, entity disjointness, value range, `rating_dist` validity,
`all_ratings` consistency, `missing_ratings_indexes` correctness, posterior array validity,
embedding shape, stats field consistency, distribution summary.

---

## Training Flags Reference

| Flag | Effect |
|---|---|
| `--transductive-learning` | Use all splits as context during training |
| `--transductive-valtest-mask` | Mask only val+test observed (train always fixed) |
| `--use-pointer` | Enable K_aug obs-obs pointer mechanism |
| `--use-graph-mask` | Hard mask: tokens with no edge cannot attend |
| `--scale-shared-rel` | Scale the shared relational bias |
| `--no-per-head-rel` | Shared relational bias (disable per-head) |
| `--use-rel-value` | Relational value augmentation |
| `--use-addone-attn` | Add-one attention (sum ≤ 1) |
| `--use-deviation-norm` | LayerNorm on deviation before adding to centroid |
| `--llm-input-dist` | Soft log-prob encoding for non-one-hot rating_dist |
| `--overwrite-existing-data` | Overwrite output dir if it exists |
| `--item-dropout-rate` | Prob of dropping item deviation (1.0 = always drop) |
| `--annotator-dropout-rate` | Prob of dropping annotator deviation |
| `--annotator-reg-weight` | L2 reg weight on annotator deviations |
| `--item-reg-weight` | L2 reg weight on item deviations |
