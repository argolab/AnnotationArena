## Trainer/Eval Renovation Plan (aligned with new DataConverter/RankingData)

Goals
- Training: input is training observed variables + training missing variables. Apply masking only to a random subset of the observed variables, then append the training missing variables. Compute loss only on observed and artificially masked entries.
- Evaluation: no masking. Input is observed + missing variables. Compute metrics and loss only on observed entries.

Key Model/Data Contracts
- RankingData now uses `status` (0=missing, 1=masked, 2=observed) and `instance` ("train"/"test"). `is_missing`, `is_masked`, `is_observed` are properties derived from `status`.
- Data comes from `DataConverter.create_variables_from_bundle(bundle, partition, status)`; variables are already 0-indexed and pairwise are clipped to `max_rank_size`.

Trainer Changes
- Simplify `ImputerTrainer.train_step` signature to accept:
  - `train_observed_vars: List[RankingData]`
  - `train_missing_vars: List[RankingData]`
  - `masking_rate: float`
- Inside `train_step`:
  - Create masked copies for a random M% of `train_observed_vars` (set `status=1`), keep the rest observed (`status=2`).
  - Build model input: `[masked_or_observed] + train_missing_vars`.
  - Forward pass and compute loss only over non-missing entries (observed+masked). No loss for missing.
- Remove support for legacy batch-of-batches; keep a minimal, clear API.

Evaluation Changes
- `EvaluationEngine.evaluate_model(model, variables, converter, device)`:
  - No masking applied. Use variables as-is.
  - Forward pass on all variables.
  - For loss/accuracy, filter to observed entries only.
  - Return breakdown keyed as `observed_metrics`. (No `masked_metrics` during eval; optionally set `missing_metrics` with counts only.)

Loss Strategy Integration
- `DefaultLossStrategy.compute(predictions, references)` already separates losses by `ref.is_masked`. During training, masked vs observed are set via `status`. During eval, all references are observed (`status=2`), so masked losses are zero.

Callback Update
- Update `EvaluationCallback` to drop `masking_rate` and `test_data`, and align with the new eval API.

Migration Notes
- Replace any constructors of `RankingData` using `is_missing`/`is_masked` args with `status` and pass through `instance`.
- Ensure no assumptions about ranking length; converter already truncates.


