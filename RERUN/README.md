# RERUN

A clean, skipjack-ready path from nothing to the paper's Domain 3 figures.

Everything here is a rewrite of `imputer/ranking/scripts/DOMAIN3-SIGMOID/`, which
spelled the experiment matrix out as 24 near-identical per-size scripts, each
hardcoding Prabhav's home directory, his email, and the old cluster's `a100`/`cpu`
partitions. Same experiments, same hyperparameters, one script per *kind* of job.

## Layout

```
RERUN/
  README.md            this file
  FIGURES.md           every paper figure -> the script that makes it
  DISCREPANCIES.md     running register of paper-vs-code mismatches
  config/cluster.sh    all site settings; edit this file and nothing else
  jobs/                one sbatch per kind of job, array-parallel over sizes
    generate_data.sbatch
    marformer_train.sbatch
    stan_fit.sbatch
    marformer_eval.sbatch
    make_llmrubric_data.sh   (not a batch job; runs in seconds)
  submit_all.sh        submits the paper's full matrix, with dependencies
  make_figures.sh      the driver: results -> figures -> Overleaf names
  runs/<RUN_ID>/       one directory per rerun (DATA, RESULTS, PLOTS, logs,
                       paper_figures, manifest.json)
```

## Quick start

```bash
cd RERUN

# see what would be submitted, without submitting
./submit_all.sh --run-id baseline --dry-run

# submit everything: data -> 35 Marformer tasks -> eval, plus 54 Stan tasks
./submit_all.sh --run-id baseline

# watch
squeue -u $USER -o '%.18i %.9P %.28j %.2t %.10M %R'

# once training is done, make the figures on a GPU node
./make_figures.sh --run-id baseline --submit

# happy with them? copy into the Overleaf clone
./make_figures.sh --run-id baseline --cache --install
```

Tweaking the architecture and rerunning is two commands, and the second one
reuses the generated data rather than regenerating it:

```bash
# edit hyperparameters in scripts/DOMAIN3/run_marformer_group.sh, then
./submit_all.sh --run-id try-L12 --data-from baseline --stage marformer
./submit_all.sh --run-id try-L12 --stage eval
```

## The matrix

`submit_all.sh` submits exactly what the paper reports. One array task per cell.

| Model | Axis | Protocol | Sizes | Tasks |
|---|---|---|---|---|
| Marformer | item | transductive | 100…1000 by 100 | 10 |
| Marformer | annot | transductive | 10…50 by 5 | 9 |
| Marformer NT | item | non-transductive | 100…900 by 100 | 9 |
| Marformer NT | annot | non-transductive | 10…40 by 5 | 7 |
| Stan Oracle | item | transductive | 100…500 by 100 | 5 |
| Stan Oracle | annot | transductive | 10…40 by 5 | 7 |
| Stan Bin / Proj / Both | item | transductive | 100…500 by 100 | 5 each |
| Stan Bin / Proj / Both | annot | transductive | 10…50 by 5 | 9 each |

35 GPU tasks, 54 CPU tasks. Stan has no non-transductive arm — that comparison
exists only for Marformer, and it is what the `_clean_nt` figures show.

## Where the knobs are

- **Cluster** (partition, walltime, memory, conda env): `config/cluster.sh`.
- **Marformer architecture and optimisation**: still
  `imputer/ranking/scripts/DOMAIN3/run_marformer_group.sh` — deliberately not
  duplicated here, so there is one source of truth. D=80, L=8, H=4, d_ff=128,
  AdamW lr=2e-4, wd=0.01, masking 15%, K=5 augmentations, masked loss weight 15,
  400 epochs.
- **Data generation** (dimensions, generative hyperparameters, split sizes):
  `jobs/generate_data.sbatch`.
- **Which sizes to run**: the `MARFORMER_MATRIX` / `STAN_MATRIX` arrays at the
  top of `submit_all.sh`.

## What changed versus the old scripts

- `--comment=accept_cost` on every submission (skipjack rejects sbatch without it).
- `h100` / `med` instead of `a100` / `cpu`; `--mem=` instead of `--mem-per-cpu=`
  (which is capped per-partition here); `--exclude=c001` dropped.
- Logs and results go to this run's directory, not to a hardcoded
  `/home/psingh54/...` path.
- The conda environment is activated **and verified**. The upstream
  `scripts/DOMAIN3/common.sh` does `conda activate prabhav2 || true`, which
  silently does nothing when that env is absent — jobs then run against
  whatever interpreter happened to be on PATH. Stan jobs additionally check
  that CmdStan is installed and fail loudly if not.
- Sizes are array tasks rather than sequential groups, so on an idle cluster the
  whole matrix finishes in roughly one task's walltime instead of the sum.

## Prerequisites

- Conda env with torch (default `torch`; override with `CONDA_ENV`).
- CmdStan, for the Stan arms only:
  `python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'`.
  Not installed on this account yet.
- `imputer/ranking/BASELINES/` — required by `plot_domain3_sigmoid.py`, which
  imports `BASELINES.structured_baselines` for the Naive Bayes / log-linear
  series. It was missing from this repo and has been copied in from `AA_new`,
  so plotting works now.

  It was previously listed in `.gitignore`, which is why a fresh clone could not
  run `plot_domain3_sigmoid.py` at all (`ModuleNotFoundError: No module named
  'BASELINES'`). That ignore rule is removed on branch `patches`, so the
  directory is now tracked.
- `imputer/ranking/scripts/convert-llm-rubric/` — the raw TSVs and converters
  that rebuild the LLM-Rubric bundles, also brought over on branch `patches`.

## Not covered here

Domains 1 and 2, the SMAR appendix, and the Logit Lens appendix. None of them
have runnable code in this repository; see the "Orphaned" table in
[FIGURES.md](FIGURES.md) for the specifics of what is missing and why.

The real-data (LLM-Rubric) arm is wired into `make_figures.sh` and its data is
regenerable -- `python scripts/convert-llm-rubric/make_llmrubric_splits.py`
rebuilds the bundles from the raw TSVs in that directory. Marformer on
LLM-Rubric still has no wrapper script; call `imputer.entity_mf.train` directly
per `scripts/README.md`.
