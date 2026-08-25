# Paper figure → script map

Every `\includegraphics` in `Marformer-Paper/main.tex`, and what produces it.
Paths in the "Produced at" column are relative to a run directory
(`RERUN/runs/<RUN_ID>/`); paths in "Overleaf" are relative to
`Marformer-Paper/`.

`make_figures.sh` regenerates everything in the **reproducible** table below and
copies it into place under the Overleaf names. Nothing in the **orphaned**
table can currently be regenerated from this repo.

---

## Reproducible: Domain 3 synthetic (DOMAIN3-SIGMOID)

All twelve come from a single script, `scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py`,
which reads `RESULTS/MARFORMER`, `RESULTS/MARFORMER-NT` and `RESULTS/STAN` and
re-runs Marformer eval on the fly.

```bash
python scripts/DOMAIN3-SIGMOID/plot_domain3_sigmoid.py --part all --calibration
```

| Paper § | Overleaf path | Produced at |
|---|---|---|
| §Results (main) | `PLOTS/DOMAIN3-SYNTH/ITEM/item_generalization_log_loss_clean_nt.png` | `PLOTS/DOMAIN3-SIGMOID/ITEM_GENERALIZATION/item_generalization_log_loss_clean_nt.png` |
| §Results (main) | `PLOTS/DOMAIN3-SYNTH/ANNOT/annot_generalization_log_loss_clean_nt.png` | `.../ANNOT_GENERALIZATION/annot_generalization_log_loss_clean_nt.png` |
| §Results (main) | `PLOTS/DOMAIN3-SYNTH/ITEM/item_generalization_mbr_l2_clean_nt.png` | `.../ITEM_GENERALIZATION/item_generalization_mbr_l2_clean_nt.png` |
| §Results (main) | `PLOTS/DOMAIN3-SYNTH/ANNOT/annot_generalization_mbr_l2_clean_nt.png` | `.../ANNOT_GENERALIZATION/annot_generalization_mbr_l2_clean_nt.png` |
| §Results (main) | `PLOTS/DOMAIN3-SYNTH/ITEM/item_calibration_size500.png` | `.../ITEM_GENERALIZATION/calibration_clean/item_calibration_size500.png` |
| §Results (main) | `PLOTS/DOMAIN3-SYNTH/ANNOT/annot_calibration_size30.png` | `.../ANNOT_GENERALIZATION/calibration_clean/annot_calibration_size30.png` |
| App. Additional Domain 3 | `PLOTS/DOMAIN3-SYNTH/ITEM/item_generalization_log_loss.png` | `.../ITEM_GENERALIZATION/item_generalization_log_loss.png` |
| App. Additional Domain 3 | `PLOTS/DOMAIN3-SYNTH/ANNOT/annot_generalization_log_loss.png` | `.../ANNOT_GENERALIZATION/annot_generalization_log_loss.png` |
| App. Additional Domain 3 | `PLOTS/DOMAIN3-SYNTH/ITEM/item_generalization_mbr_l2.png` | `.../ITEM_GENERALIZATION/item_generalization_mbr_l2.png` |
| App. Additional Domain 3 | `PLOTS/DOMAIN3-SYNTH/ANNOT/annot_generalization_mbr_l2.png` | `.../ANNOT_GENERALIZATION/annot_generalization_mbr_l2.png` |
| App. Additional Domain 3 | `PLOTS/DOMAIN3-SYNTH/ITEM/item_generalization_runtime.png` | `.../ITEM_GENERALIZATION/item_generalization_runtime.png` |
| App. Additional Domain 3 | `PLOTS/DOMAIN3-SYNTH/ANNOT/annot_generalization_runtime.png` | `.../ANNOT_GENERALIZATION/annot_generalization_runtime.png` |

**Calibration variant.** The script emits two calibration directories.
`calibration_clean` plots Marformer + Marformer NT + the four Stan models
(`CLEAN_NT_SERIES`); `calibration_all` adds the empirical baselines but drops
Marformer NT. The main-text calibration figure sits beside the `_clean_nt`
curves, so `make_figures.sh` ships `calibration_clean`. Override with
`--calibration-variant calibration_all`.

**Series in the `_clean_nt` figures** are Marformer, Marformer NT, Stan Oracle,
Stan Both, Stan Bin, Stan Proj. The non-`clean` variants add Unigram Pool ij,
Naive Bayes IJK, Structured NB and Structured Log-Linear — these come from
`BASELINES/structured_baselines/`, fitted inside the plot script, not from a
separate training run.

## Reproducible: Domain 3 real data (LLM-Rubric)

```bash
python scripts/LLMRubric/plot_llmrubric_cpm_sigmoid.py --calibration
```

| Paper § | Overleaf path | Produced at |
|---|---|---|
| §Real Annotation Data | `PLOTS/REAL-LLMRUBRIC/llmrubric_cpm_log_loss.png` | `PLOTS/LLMRUBRIC-CPMSIGMOID/ITEM_GENERALIZATION/llmrubric_cpm_log_loss.png` |
| §Real Annotation Data | `PLOTS/REAL-LLMRUBRIC/llmrubric_cpm_mbr_l2.png` | `.../llmrubric_cpm_mbr_l2.png` |
| §Real Annotation Data | `PLOTS/REAL-LLMRUBRIC/llmrubric_cpm_calibration_size175.png` | `.../calibration/llmrubric_cpm_calibration_size175.png` |
| §Real Annotation Data | `PLOTS/REAL-LLMRUBRIC/llmrubric_cpm_calibration_size10.png` | `.../calibration/llmrubric_cpm_calibration_size10.png` |
| App. Additional Domain 3 (Real) | `PLOTS/REAL-LLMRUBRIC/llmrubric_cpm_runtime.png` | `.../llmrubric_cpm_runtime.png` |

**Data is regenerable.** `scripts/README.md` says these bundles have no
generation script and to *"ask the repo owner"* -- that is out of date. The
converters and the raw TSVs now live in `scripts/convert-llm-rubric/`
(brought over from `AA_new` on branch `patches`):

```bash
cd imputer/ranking
python scripts/convert-llm-rubric/make_llmrubric_splits.py
```

builds `DATA/LLM_RUBRIC/LLMRubric_225_25_9_{10,50,100,150,175}/` from
`human_judges_real_convs_FIXED_ANON.tsv` and
`gpt-3.5-turbo-16k_real_evaluations_FIXED.tsv` (seed 42; 175 train / 25 val /
25 test items; nested training subsets). Both calibration sizes the paper uses
(10 and 175) are covered. To add the intermediate sizes the old README lists
(20, 30, 40, 75, 125), extend `TRAIN_SIZES` at the top of that script.

`make_figures.sh` still skips this section with an explanation if the data is
absent, rather than failing.

**Path mismatch, still unfixed upstream.** `scripts/LLMRubric/CPMSigmoid/run_size*.sh`
writes Stan results to `RESULTS/STAN/LLM_RUBRIC/CPMSigmoid`, but
`plot_llmrubric_cpm_sigmoid.py` reads `RESULTS/STAN/CPMSigmoid` (line 117).
`make_figures.sh` bridges this with a compatibility symlink; if you run the
plot script by hand, create it yourself.

---

## Orphaned: no generating code in this repo

| Overleaf path | Paper § | Situation |
|---|---|---|
| `PLOTS/DOMAIN1/KLCurves-MCAR-DOM1.png` | Domain 1 results | `imputer/graphical_synthetic_modeling/` writes generic filenames (`attention_summary_plots.png`, …) and was renamed by hand. Frozen since 2026-04-11. |
| `PLOTS/DOMAIN1/Scatter-MCAR-DOM1.png` | Domain 1 results | same |
| `PLOTS/DOMAIN1/TrainingCurve-DOM1.png` | Domain 1 runtime | same |
| `PLOTS/DOMAIN2/KLCurves-MCAR-DOM2.png` | Domain 2 results | Domain 2 run scripts are Haojun's and were never committed — `main.tex` says so explicitly. |
| `PLOTS/DOMAIN2/Scatter-MCAR-DOM2.png` | Domain 2 results | same |
| `PLOTS/DOMAIN2/TrainingCurve-DOM2.png` | Domain 2 runtime | same |
| `PLOTS/DOMAIN1/KLCurves-SMAR-DOM1.png` | App. MAR setting | `SMAR` appears in **no file** in either repo. |
| `PLOTS/DOMAIN1/Scatter-SMAR-DOM1.png` | App. MAR setting | same |
| `PLOTS/DOMAIN2/KLCurves-SMAR-DOM2.png` | App. MAR setting | same |
| `PLOTS/DOMAIN2/Scatter-SMAR-DOM2.png` | App. MAR setting | same |
| `PLOTS/DOMAIN3-SYNTH/LENS/layerwise_metrics.png` | App. Logit Lens | No `lens` / `layerwise` / tuned-probe code exists. The appendix quotes exact numbers (log-loss 1.386 → 0.566, smECE 0.163 → 0.027), so the analysis was run — the script was never committed. |
| `PLOTS/DOMAIN3-SYNTH/LENS/selected_calibration.png` | App. Logit Lens | same |

Jason's open asks in `main.tex` that need code that does not exist yet: MALA
curves for Domain 2 (and ideally Domain 1), extending the Domain 1/2 x-axis the
way Domain 3 was extended, and renaming the baseline legends to
`MAP (EM)` / `MAP (SGD)` / `MAP (SGD -- true init)`.

## Hand-drawn (no script, do not regenerate)

`PLOTS/Figure1.png` (currently commented out), `PLOTS/Marformer-TrainEval.png`,
`PLOTS/MarformerArchitecture.png`, `PLOTS/MarformerBlockAttn.png`.
`PLOTS/Marformer-StructName-Arch.png` and everything under `PLOTS/OLD/` are
unreferenced by `main.tex`.

---

## Duplicated includes

`main.tex` includes the six main-text Domain 3 figures **twice** — once in
§Experiments and Results (lines ~1592–1639) and again in
§"PLACEHOLDER for old material" (lines ~1755–1795). The placeholder section is
marked for deletion once its text is merged. Same files either way, so
regeneration is unaffected.

---

## Paper vs. code discrepancies

Tracked separately, and meant to be revisited before each rerun and at
paper-finalisation time: [DISCREPANCIES.md](DISCREPANCIES.md).
