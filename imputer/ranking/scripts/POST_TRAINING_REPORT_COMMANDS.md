# Commands after training finishes

Run from **`imputer/ranking`** (replace the `cd` path if yours differs).

```bash
cd ~/AA_new/imputer/ranking
export PYTHONPATH=.
```

---

## SummEval (part B: 750 / 1000 / 1280)

**GPU — ranked_eval for all three runs + text summaries + PNGs (tables + lineplot):**

```bash
bash scripts/SUMMEVAL/MARFORMER/TRAIN/eval_train_b_750_1000_1280_tmp.sh
```

**GPU — single run:**

```bash
python -u -m imputer.entity_mf.ranked_eval \
  --run-dir RESULTS/MARFORMER/SUMMEVAL/SummEval_1600_8_4_1280 \
  --ranks 1,3,5,7 --device cuda
```

**CPU only — regenerate vertical tables + `ranked_eval_vertical.png` + `val_xent_k1_vs_train_size.png` (needs JSON already):**

```bash
python -m imputer.entity_mf.ranked_eval_report --mode summeval
```

**Slurm example:**

```bash
cd ~/AA_new/imputer/ranking
PARTITION=h100 GPUS=1 TIME=02:00:00 CPUS_PER_TASK=4 MEM_PER_CPU=8G \
  /home/xwang397/bin/sbatch_adapt scripts/SUMMEVAL/MARFORMER/TRAIN/eval_train_b_750_1000_1280_tmp.sh
```

**Outputs (under `RESULTS/MARFORMER/SUMMEVAL/`):**

- `<run>/RANKED_RESULTS/by_val_missing_xent.json`
- `reports/ranked_eval_vertical.png`
- `reports/val_xent_k1_vs_train_size.png` — val missing xent at k=1 vs training size; **filled** markers = training reached `training.epochs` in `train_config.json`, **hollow** = stopped early (incomplete). If `training_history.json` lacks the k=1 checkpoint epoch, the y-value comes from the ranked JSON (no SD band for that point).

---

## STAN (22 runs, 4 families)

**GPU — one family (lighter job):**

```bash
bash scripts/STAN/eval_stan_marformer_22_tmp.sh 1   # Factor_250 AnnotatorTest (5)
bash scripts/STAN/eval_stan_marformer_22_tmp.sh 2   # Factor_650 ItemTest (6)
bash scripts/STAN/eval_stan_marformer_22_tmp.sh 3   # Normal_250 AnnotatorTest (5)
bash scripts/STAN/eval_stan_marformer_22_tmp.sh 4   # Normal_650 ItemTest (6)
```

**GPU — all 22 ranked_evals + summaries + PNGs:**

```bash
bash scripts/STAN/eval_stan_marformer_22_tmp.sh
```

**CPU only — tables + lineplots (JSON must exist):**

```bash
bash scripts/STAN/eval_stan_marformer_22_tmp.sh summary
```

**Or:**

```bash
python -m imputer.entity_mf.ranked_eval_report --mode stan
```

**Slurm example (one family):**

```bash
cd ~/AA_new/imputer/ranking
PARTITION=a100 GPUS=1 TIME=2:00:00 CPUS_PER_TASK=4 MEM_PER_CPU=8G \
  /home/xwang397/bin/sbatch_adapt scripts/STAN/eval_stan_marformer_22_tmp.sh 2
```

**Outputs (under `RESULTS/MARFORMER/STAN/`):**

- `<run>/RANKED_RESULTS/by_val_missing_xent.json`
- `reports/ranked_vertical_<family>.png` (4)
- `reports/val_xent_k1_vs_train_size_<family>.png` (4) — same marker convention as SummEval (filled = complete training, hollow = incomplete; y from history when possible, else ranked JSON)
- `reports/val_xent_k1_four_curve_STAN.png` — two panels (Item-test vs Annotator-test), Factor vs Normal

---

## Optional flags

```bash
python -m imputer.entity_mf.ranked_eval_report --mode stan --no-png          # text only
python -m imputer.entity_mf.ranked_eval_report --mode stan --no-lineplot      # skip lineplots, keep table PNGs
python -m imputer.entity_mf.ranked_eval --run-dir RESULTS/MARFORMER/STAN/<RUN> --no-last
```

**Lineplots only (needs `by_val_missing_xent.json` per run; `training_history.json` optional but improves the SD band when present for the k=1 epoch):**

```bash
python -c "from pathlib import Path; from imputer.entity_mf.ranked_eval_lineplot import write_all_lineplots_summeval, write_all_lineplots_stan; write_all_lineplots_summeval(Path('RESULTS/MARFORMER/SUMMEVAL')); write_all_lineplots_stan(Path('RESULTS/MARFORMER/STAN'))"
```

---

## Train jobs (reference)

```bash
bash scripts/STAN/stan_data_command_marformer.sh
```

SummEval part B: `scripts/SUMMEVAL/MARFORMER/TRAIN/run_train_b_{750,1000,1280}.sh` (submit similarly with `sbatch_adapt`).
