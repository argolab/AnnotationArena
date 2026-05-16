# Recurrent Entity Marformer

Weight-shared recurrent variant of Entity Marformer, kept separate from the flat model and training scripts.

## Architecture

```
prelude (prelude_depth unique blocks)
  -> core (num_core_layers blocks, applied num_recurrence times, shared weights)
  -> coda (coda_depth unique blocks)
```

**Effective depth** (compare to flat `--num-layers`):

```
effective_depth = prelude_depth + num_core_layers * num_recurrence + coda_depth
```

Example matching flat `num_layers=8` (DOMAIN3-FINAL scripts use this):

- `prelude_depth=1`, `num_core_layers=2`, `num_recurrence=3`, `coda_depth=1` → effective depth 8

## Commands

From `imputer/ranking` with `export PYTHONPATH=.`:

```bash
# Train
python -u -m imputer.entity_mf.recurrent.train \
  --data-dir DATA/STAN/DOMAIN3-FINAL/ItemSplits/Transductive/DOMAIN3-FINAL_Item_T_100 \
  --run-name DOMAIN3-FINAL_Item_T_100_RECURRENT_MF \
  --output-root RESULTS/RECURRENT_MARFORMER/STAN/DOMAIN3-FINAL \
  --prelude-depth 1 --num-core-layers 2 --num-recurrence 3 --coda-depth 1 \
  --embedding-dim 80 --epochs 400 --transductive-learning --use-pointer --no-per-head-rel --scale-shared-rel

# Test
python -u -m imputer.entity_mf.recurrent.test \
  --run-dir RESULTS/RECURRENT_MARFORMER/STAN/DOMAIN3-FINAL/DOMAIN3-FINAL_Item_T_100_RECURRENT_MF
```

## Shell scripts

`scripts/DOMAIN3-FINAL/RecurrentMarformer/ItemExpansion/run_train.sh` — single bundle `DATA/DOMAIN3-OLD_Item_T_1000` (default **400** epochs).  
`run_recurrence_sweep.sh` — several `(prelude, core, recurrence, coda)` tuples sequentially, also **400** epochs by default. Active sweep configs target **8 unique blocks** (`prelude + core + coda`) to match flat Marformer param count, with **actual depth** `prelude + core × recurrence + coda` **> 8** (deeper forward pass via core weight sharing). Writes to `RESULTS/RECURRENT_MARFORMER/DOMAIN3-OLD-UNIQUE8-DEEP/` (not `DOMAIN3-OLD/`). Older effective-depth-8 tuples are kept commented in the script.  
`run_eval_sweep.sh` — eval first sweep under `DOMAIN3-OLD/`.  
`run_eval_sweep_unique8_deep.sh` — eval the UNIQUE8-DEEP sweep.  
`run_recurrence_sweep_p0c1rx.sh` — thin-core sweep `(0, 1, x, 0)` for `x ∈ {6,8,10,12,14,16}` → `DOMAIN3-OLD-P0C1RX/`.  
`run_eval_sweep_p0c1rx.sh` — eval that sweep.  
`run_recurrence_sweep_unique12.sh` — **12 unique blocks** (`prelude + core + coda = 12`), varied splits → `DOMAIN3-OLD-UNIQUE12/`.  
`run_eval_sweep_unique12.sh` — eval that sweep.  
`run_recurrence_scaling.sh` — load one run’s checkpoint and sweep `num_recurrence` at **test** time (see `recurrence_scaling_eval.py`).  
`run_recurrence_scaling_p1c2r3c1.sh` / `run_recurrence_scaling_p0c2r4c0.sh` — presets for those runs.  
`run_recurrence_scaling_sweep_domain3_old.sh` — same eval for every `DOMAIN3-OLD_Item_T_1000_RECURRENT_MF_*` run.  
`run_eval_test.sh` — test eval for the default `p1c2r3c1` run.  
`run_eval_sweep.sh` — eval all sweep runs.

## Results layout

| Flat Marformer | Recurrent |
|----------------|-----------|
| `RESULTS/MARFORMER/...` | `RESULTS/RECURRENT_MARFORMER/...` |
| `imputer.entity_mf.train` | `imputer.entity_mf.recurrent.train` |

`train_config.json` includes `model_type: recurrent_entity_marformer` and the four depth fields plus `effective_depth`.

## Shared code

- `entity_mf/blocks.py` — transformer block build/forward
- `entity_mf/backbone.py` — embeddings, streams, pointer cache
- `entity_mf/lightning_module.py` — training loop (used by both trainers)
- `entity_mf/eval.py` — evaluation (unchanged)
