# Repository Guidelines

## Project Structure & Module Organization
- Core Python code lives in `imputer/`, with active model code in `imputer/entity_mf/` (`train.py`, `test.py`, `model.py`, `data.py`, `eval.py`).
- Baselines are in `imputer/baseline_mlp/`.
- Experiment and utility scripts live in `scripts/` (for example, `scripts/SAMPLE_SCRIPTS/`, `scripts/TESTING_SCRIPTS/`, `scripts/utils/`).
- Dataset inputs are typically under `DATA/`; generated artifacts and run outputs are written to `OUTPUT/` and `RESULTS/`.
- STAN-related resources and commands are in `STAN/` and `scripts/stan/`.

## Build, Test, and Development Commands
- Set local import path before running modules: `export PYTHONPATH=.`
- Train Entity Marformer locally:
  - `python -u -m imputer.entity_mf.train --data-dir <bundle_dir> --epochs 50 --device cuda --output-root OUTPUT/ENTITY_MF`
- Evaluate a trained run (best/last checkpoints):
  - `python -u -m imputer.entity_mf.test --run-dir <run_dir> --checkpoint both --device cuda`
- Run batch evaluation helpers:
  - `bash scripts/TESTING_SCRIPTS/run_test_summeval.sh`
  - `bash scripts/TESTING_SCRIPTS/run_test_llmrubric.sh`
- Cluster/STAN entrypoint example: `bash run.sh` (submits inference workflow).

## Coding Style & Naming Conventions
- Use Python 3.10+ conventions, 4-space indentation, and type hints for new/modified functions.
- Keep modules focused and composable (data conversion, model, eval, masking, training are intentionally separated).
- Follow existing naming patterns: `snake_case` for functions/variables, `PascalCase` for classes, uppercase for constants.
- Prefer short docstrings describing inputs/outputs and avoid adding unrelated refactors in feature/fix PRs.

## Testing Guidelines
- This repo primarily uses script-based validation rather than a full unit-test suite.
- For model changes, run at least one train smoke test and one `imputer.entity_mf.test` evaluation on a known run directory.
- For data pipeline changes, validate bundle integrity with `python test_data/check_bundle.py`.
- Keep test/eval runs reproducible by recording `--data-dir`, key hyperparameters, and checkpoint choice.

## Commit & Pull Request Guidelines
- Recent history uses short, descriptive subjects (for example, `Tensor Scripts`, `Discrete`); keep commit titles concise and specific.
- Prefer imperative, scope-aware messages (example: `entity_mf: fix masked loss aggregation`).
- PRs should include: purpose, changed paths, exact commands run, and before/after metrics when behavior changes.
- Link related issue(s) when available and attach plots/screenshots for result or visualization updates.

## Security & Configuration Tips
- Do not commit large generated artifacts, secrets, or machine-specific absolute paths.
- Keep environment assumptions explicit (see `OUTPUT/env.yaml`) and document GPU/CPU expectations in scripts.
