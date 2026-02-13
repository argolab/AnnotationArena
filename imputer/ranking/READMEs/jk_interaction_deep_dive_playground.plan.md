---
name: JK interaction deep dive (playground)
overview: Working plan for JK-only (hold_I_constant=1). Prioritize diagnostics (1C) first; defer topology-controlled missingness (4) until we have evidence from 1C that topology is a main driver of the Stan↔Marformer gap.
isProject: false
---

# JK Interaction Deep Dive (Playground)

## Current stance / assumptions

- Stan encodes the domain likelihood + latent structure; Marformer is learning it implicitly from limited observations, so a Stan > Marformer gap is not surprising a priori.
- Before investing in “hard” missingness samplers, we want **measurable evidence** from diagnostics that JK failures correlate with graph/topology pathologies.
- This plan is written to be handed off: each subsection specifies (a) inputs, (b) outputs, and (c) decision criteria.

## Section 1 — Missingness Ladder + JK Bipartite Diagnostics (do first)

### 1A) Missingness ladder (easy → harder), *excluding* MNAR for now

We’ll start with patterns we can implement via bundle rewrite and that are likely to be interpretable:

1) **MCAR-IID**
   - Definition: randomly choose an observed subset of rating entries uniformly over `(i,j,k)` at target missing rates `{0.1, 0.3, 0.5, 0.7}`.
   - Why: baseline “is the task intrinsically learnable for Marformer when observation is IID?”
2) **Balanced-degree MCAR**
   - Definition: enforce minimum training-observed degree constraints (e.g., each annotator appears in ≥m observed ratings, each item appears in ≥n observed ratings), then sample additional edges uniformly until reaching the target observed fraction.
   - Why: removes trivial pathologies (isolates / near-isolates) while still being “random enough”.
3) **Row/col structured holes**
   - Annotator-block:
     - Definition: choose a subset of annotators and hide most of their ratings (in train and/or test), while keeping other annotators relatively well-observed.
     - Purpose: simulate “cold-start annotator” / “long-tail annotator” regimes.
   - Item-block:
     - Definition: choose a subset of items and hide most ratings touching those items.
     - Purpose: simulate “cold-start item” regimes.
   - Note: keep the complement reasonably observed so failures are attributable to the chosen hole structure.
4) **Connectivity-controlled** (topology ablations)
   - Status: *defer implementation* (see below); we only add this after 1C demonstrates topology is predictive of failure.
5) **Tie-breaking-like / sequential MAR**
   - What we currently have.
  
### 1B) Bundle rewrite “service” (enabler for 1A)

Instead of re-running Stan generation for each pattern, create a script that:

- loads an existing dataset directory containing:
  - `data_bundle.json` (expects `all_ratings` as the full universe),
  - `configs.json` (used for sizes and reproducibility bookkeeping).
- overwrites `observed_ratings` / `missing_ratings` according to a chosen pattern and seed,
- (optional for later) can also rewrite `observed_pairwise` / `missing_pairwise`, but ratings-only is sufficient for JK diagnosis,
- updates `stats` consistently:
  - total/observed/missing counts,
  - train vs test observed rates,
  - embed protocol metadata (e.g., `stats["rewrite_protocol"]`, `stats["rewrite_params"]`, `stats["rewrite_seed"]`),
- writes a new dataset directory:
  - Use nice naming that shows the missingness pattern modification that is both readable and concise.
  - copy-through `configs.json`,
  - write updated `data_bundle.json`.
Then both domain_model.stan and our marformer/imputer model is able to train on that new data.

Inputs:
- `INPUT_DIR` pointing at a base run produced by `stan/scripts/generate_data.py` (e.g., JK mode from `scripts/easy_data/easy_JK.sh`).
- `pattern` + `pattern_params` + `seed`.

Outputs:
- `OUTPUT_DIR` containing a rewritten `data_bundle.json` that can be consumed unchanged by:
  - `imputer/run_imputer.py` and
  - `stan/scripts/run_inference.py` + `stan/scripts/evaluate_predictions.py`.

Acceptance checks:
- `imputer/data.py:DataConverter.validate_bundle` passes on the rewritten bundle.
- Observed+missing partitions are disjoint and cover `all_ratings`.
- `stats` fields match the realized counts (not just the requested rate).

### 1C) JK bipartite diagnostics infra (highest leverage)

Add a post-run analysis step that can be run on *any* dataset/pattern and produces:

- **JK matrices** on **train-missing** ratings: (Note we always just engineer and care about train instance)
  - `JK_connectedness[j,k]`: binary connectedness, 1 if that jk element is observed, 0 if missing.
  - `JK_logloss[j,k]`: average negative log prob assigned to the true rating on entries with annotator=j and item=k.
  - `JK_acc[j,k]`: average 0/1 correctness on those entries.
  - `JK_rmse[j,k]`: RMSE of predicted rating vs true rating on those entries (use the same rating scale as the rest of the codebase, typically 1..C; record the convention in `jk_diagnostics.json`).
  - Implementation detail: aggregate across attributes `i` since `hold_I_constant=1` (i is invariant); still keep counts so sparse cells are handled safely.
- **Low-rank structure probes**:
  - For each metric matrix `M ∈ {JK_logloss, JK_acc, JK_rmse}` fit a **rank-1 approximation** `M ≈ u ⊗ v` (outer product), with consistent NaN-handling (SVD on a filled/weighted matrix or ALS; solver choice is less important than stable, comparable outputs).
  - **Primary visualization requirement (one-to-one with each original metric matrix):**
    - For each metric `M`, produce a “rank-1 summary plot” consisting of:
      1) a heatmap of the original matrix `M` (same colormap/range conventions across runs),
      2) an x-axis histogram/line plot for the fitted item factor `v` (length K), aligned with the K axis,
      3) a y-axis histogram/line plot for the fitted annotator factor `u` (length J), aligned with the J axis.
    - Interpretation (state this explicitly in captions):
      - Under the separability assumption (“J and K contribute independently”), `u` and `v` can be read as **proxy contribution vectors** for how much each annotator/item contributes to the metric:
        - `JK_logloss`: higher `u_j` / `v_k` suggests higher difficulty/uncertainty contributions (up to scale),
        - `JK_rmse`: higher `u_j` / `v_k` suggests larger error-magnitude contributions,
        - `JK_acc`: higher `u_j` / `v_k` suggests higher correctness contributions (less linear, but still a useful summary).
      - This is especially useful when we fit rank-1 factors to *topology-derived matrices* too (e.g., a `JK_connectedness` proxy): the resulting `u`/`v` become interpretable proxies for “how connected each annotator/item is” under the rank-1 assumption.
  - Companion: a residual heatmap `M - (u⊗v)` (same dimensions) to highlight structured deviations from rank-1.
- **Graph topology summaries** derived from **train-observed** edges:
  - Bipartite graph between annotators J and items K (edge exists if any rating observed for that `(j,k)`; optionally weight edges by count across attributes).
  - Per-node degrees: `deg(j)`, `deg(k)`; histograms and quantiles.
  - Connectivity: number of connected components, giant component fraction, isolates.
- **Correlation analyses** linking topology → predictive performance:
  - For annotators: correlate `deg(j)` with avg test-missing logloss/acc over all items for that annotator.
  - For items: correlate `deg(k)` with avg test-missing logloss/acc over all annotators for that item.
  - Report Pearson + Spearman, plus bucketed plots (e.g., degree deciles).
  - Deliverable plots (standard set so we can compare across datasets):
    - Scatter plots:
      - `deg(j)` vs `avg_logloss(j)`, `avg_rmse(j)`, `avg_acc(j)`
      - `deg(k)` vs `avg_logloss(k)`, `avg_rmse(k)`, `avg_acc(k)`
      - Include a best-fit line (or LOWESS) and annotate Pearson/Spearman on the plot.
    - Binned trend plots:
      - degree deciles on x-axis, mean metric on y-axis, with error bars (std/sem) and counts per bin.

Inputs:
- Dataset bundle: `.../data_bundle.json` (to compute train-observed topology and to know what is missing).
- Marformer outputs: `.../predictives.json` from `imputer/run_imputer.py` (to compute per-entry logloss/acc).

Standard outputs (stable filenames so sweeps are easy to diff):
- `jk_diagnostics.json` (summary numbers + correlations + counts)
- `jk_matrix_logloss.csv`, `jk_matrix_acc.csv`, `jk_matrix_rmse.csv` (+ optional `jk_matrix_count.csv`)
- `plots/`:
  - `jk_heatmap_connected.png`, `jk_heatmap_logloss.png`, `jk_heatmap_acc.png`, `jk_heatmap_rmse.png`
  - `jk_rank1_approx_logloss.png`, `jk_rank1_approx_logloss_residual.png`
  - `jk_rank1_approx_acc.png`, `jk_rank1_approx_acc_residual.png`
  - `jk_rank1_approx_rmse.png`, `jk_rank1_approx_rmse_residual.png`
  - `degree_vs_logloss_annotator.png`, `degree_vs_logloss_item.png`
  - `degree_vs_acc_annotator.png`, `degree_vs_acc_item.png`
  - `degree_vs_rmse_annotator.png`, `degree_vs_rmse_item.png`
  - `degree_binned_vs_metric_annotator.png`, `degree_binned_vs_metric_item.png`

Decision criteria (what 1C should tell us):
- If most error mass is concentrated on low-degree nodes and degree strongly predicts logloss, missingness/topology is plausibly the root cause → prioritize missingness/batching/density fixes.
- If error is spread uniformly and rank-1 probes show weak structure, the gap is less about topology → prioritize optimization/inductive bias earlier.
- If errors cluster into a small set of annotators or items (outliers), consider targeted interventions (e.g., reweighting, curricula, or embedding regularization).

This infra is the “decision engine” for what to implement next (including whether topology-controlled masks are worth it).

### Deferred: connectivity-controlled missingness (pattern 4)

We explicitly **push topology-controlled mask generation to later**.

Trigger to revisit:

- 1C shows strong evidence that topology features predict failure (degree/connectivity correlations, component pathologies, clear bottleneck signatures), *and*
- simpler ladder patterns (1–3, 5) do not already give a controlled way to stress that exact feature.

When revisited, “connectivity-controlled” should be defined as concrete samplers with explicit knobs (e.g., number of components, bridge edges, degree skew) that preserve overall observed fraction; but we only build that after 1C tells us which knobs matter.

## Section 2 — Density / safe curricula (no model structure changes)

**Objective**: test whether the JK gap is primarily an *optimization / sample-efficiency* problem (i.e., Marformer can represent the solution but doesn’t reach it reliably under the target missingness), without changing the model class.

### 2A) Curriculum and “drift” tests (what question each answers)

We keep only curricula that map cleanly to a hypothesis and have a non-leaky evaluation.

1) **Safe curriculum: training-time masking schedule (same fixed dataset)**
   - Mechanism: keep the dataset’s observed/missing split fixed; only change training-time masking of *already-observed* entries over epochs (e.g., `masking_rate` schedule, `mask_augmentations` schedule, masked/observed loss-weight schedules).
   - Question it answers: “Is the target JK regime hard because optimization is brittle early (needs an easier starting objective), but once it finds structure it can succeed?”
   - Expected signatures:
     - If schedules close the Stan gap and stabilize training variance across seeds → optimization path / curriculum matters.
     - If schedules don’t help, and diagnostics (1C) show topology pathologies dominate → missingness/topology rather than optimization path.

2) **Non-leaky pretrain→mask “drift” test (same underlying instance, but never evaluate on revealed entries)**
   - Mechanism: split the universe into three disjoint sets:
     - `S_train_obs_base`: entries always observed during all phases,
     - `S_holdout_never_seen`: entries never used for supervision (final evaluation target),
     - `S_curriculum_extra`: entries temporarily used *only in Phase 0* to create a good initialization, then removed (become treated as missing thereafter).
   - Run:
     - Phase 0 (pretrain): train with `S_train_obs_base ∪ S_curriculum_extra` as observed.
     - Phase 1 (target): continue training with only `S_train_obs_base` observed; treat `S_curriculum_extra` as missing inputs.
     - Evaluate throughout on `S_holdout_never_seen` (and optionally track loss on `S_curriculum_extra` as a “drift probe” but do not use it to select checkpoints).
   - Questions it answers:
     - “If we start from a good predictive state (by extra supervision), does training on the target missingness *preserve* the solution or does it drift away?”
     - “Is the gap caused by inability to *learn* the structure, or inability to *maintain* it under the target objective?”
   - Interpretation:
     - If Phase 0 improves holdout and Phase 1 preserves it → learning is possible; target regime is mostly an initialization / sample-efficiency issue.
     - If Phase 1 degrades holdout (while training objective improves) → objective mismatch / representation of missingness, not just init.

3) **Pretrain on different instances (transfer)**
   - Mechanism: pretrain Marformer on dense/MCAR datasets generated from different seeds (different full rating matrices), then fine-tune on the target sparse structured dataset.
   - Question it answers: “Is the JK difficulty mostly about general training dynamics (model needs generic ‘JK factor’ skills), or is it instance-specific identifiability driven by observation topology?”
   - Interpretation:
     - If transfer helps across many targets → suggests missingness-induced sample inefficiency more than intrinsic identifiability.
     - If transfer doesn’t help and 1C shows strong degree/connectivity effects → suggests topology/observational design is the bottleneck.

### 2B) Density-boost knobs (mechanisms, not new assumptions)

- **Masking-rate schedules**: e.g., low masking early → target masking late; or inverse schedule if you suspect strong denoising is needed early.
- **Mask augmentation schedules**: increase number of masking patterns over time to expand effective supervision.
- **Larger pools + subset-forward**: generate larger `(J,K)` but sample induced subgraphs per step (bounded compute) to increase diversity and co-occurrence coverage across steps.
- **Loss-weight schedules**: reweight masked vs observed loss (and pairwise loss if used) to shape the optimization landscape.

## Section 3 — Inductive bias (model changes)

Only after 1C + Section 2 give a clear story:

- Bilinear JK auxiliary head + ordinal thresholds
- Deterministic binning / ordinal consistency auxiliary loss
- Low-rank regularization / hybrid head

## Experiment catalog (grouped)

- G1: Bundle rewrite missingness ladder
  - G1.1 MCAR-IID sweep
  - G1.2 Balanced-degree MCAR sweep
  - G1.3 Annotator-block holes sweep
  - G1.4 Item-block holes sweep
  - G1.5 Tie-breaking-like rewrite (approximate protocol)
  - G1.X Connectivity-controlled masks (DEFERRED; gated by 1C findings)
- G2: JK bipartite diagnostics infra
  - G2.1 JK acc/logloss matrices (test-missing)
  - G2.2 Rank-1 approx + residual
  - G2.3 Degree/connectivity summaries + correlations
  - G2.4 Standardized outputs (json/csv/plots)
- G3: Density / safe curricula (training-only)
  - G3.1 Masking-rate schedule
  - G3.2 Mask-augmentation schedule
  - G3.3 Larger pool + subset-forward
  - G3.4 Loss-weight schedules
- G4: Inductive bias (model changes)
  - G4.1 Bilinear JK head + ordinal thresholds
  - G4.2 Deterministic binning auxiliary loss
  - G4.3 Low-rank regularizer / hybrid head

## Suggested execution order (minimal dependencies)

1) Implement G2 (diagnostics) first, because it is read-only on existing outputs and informs everything else.
2) Implement G1 (bundle rewrite) next, because it produces controlled datasets for G2 to analyze.
3) Then run a small sweep on {G1.1, G1.3, G1.5} to identify “designed failure cases”.
4) Only then decide whether to invest in:
   - topology-controlled missingness samplers (deferred 1A.4),
   - density curricula (G3),
   - or inductive bias (G4).

## Research questions map (RQ → experiments → decision rules)

- G1
  - RQ: “Does Marformer’s JK gap appear only under structured missingness (vs IID)?”
    - If gap disappears under MCAR-IID but appears under tie-breaking-like rewrites → missingness structure/selection is driving difficulty; prioritize missingness-aware training or rewrite patterns that match production.
    - If gap persists even under MCAR-IID at moderate missing rates → not just structure; consider optimization (G3) or inductive bias (G4) earlier.
  - RQ: “Is the gap explained by cold-start slices (rows/cols missing)?”
    - If annotator-block or item-block sharply worsens metrics concentrated on that slice → failure is identifiability for long-tail nodes; prioritize density/batching/schedules targeted at degree imbalance (and document the regime clearly).

- G2
  - RQ: “Are JK errors explained by graph topology (degree/connectivity/bottlenecks)?”
    - If degree strongly predicts logloss and errors concentrate on low-degree nodes → topology is a main driver; consider investing in topology-controlled masks *only if* you need cleaner stress tests beyond G1.
    - If errors are not explained by topology and rank-1 probes show weak structure → the gap is less about observational identifiability; prioritize optimization dynamics (G3) or inductive bias (G4).
  - RQ: “Is the failure approximately low-rank (u_j × v_k) or more complex?”
    - If rank-1 (or low-rank) explains most variation in JK logloss → a bilinear/ordinal bias (G4) is a high-priority candidate.
    - If residuals are structured (clusters/outliers) → consider targeted interventions (reweighting, per-annotator calibration, or diagnostics for specific annotators/items).

- G3
  - RQ: “Is JK primarily an optimization/sample-efficiency problem?”
    - If masking schedules / augmentation schedules close the gap without changing the model → optimization path is the issue; keep model class and focus on training recipe.
    - If pretrain→mask drift test shows degradation during Phase 1 → objective mismatch or missingness handling is causing drift; consider changing objectives or adding bias (G4).
  - RQ: “Does generic JK pretraining transfer to sparse structured targets?”
    - If transfer helps broadly → the model benefits from learning generic latent-factor structure; invest in pretraining pipelines.
    - If transfer does not help and 1C shows topology effects → observational identifiability dominates; focus on observation patterns / data collection assumptions.

- G4
  - RQ: “Is the missing ingredient explicit JK factorization + ordinal structure?”
    - If a bilinear+threshold head reduces the gap (especially where rank-1 probes predict) → inductive bias is warranted; iterate on the minimal bias.
    - If it doesn’t help and 1C indicates topology dominates → bias won’t fix underidentification; prioritize data/missingness regimes or training-time constraints instead.
