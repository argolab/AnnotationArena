# Domain 3 replication on freshly generated data

Run: `RERUN/runs/replicate-01`. Everything below was produced from scratch —
new synthetic data from the Stan CPM generator (seed 42), 38 Marformer
trainings, and the paper's own plotting scripts. No cached results from the old
cluster were reused for the synthetic arm.

**Verdict: the Domain 3 results replicate.** Calibration matches within
confidence intervals, the real-data log-loss matches to ~0.02 nats, and the
qualitative claims hold. Two things needed fixing to get there, both recorded in
[DISCREPANCIES.md](../DISCREPANCIES.md).

---

## 1. Data

`jobs/generate_data.sbatch` regenerates the base bundle
(`K_train=4800, I=9, J=50, C=4`, MCAR at 50%) and derives all 38 splits. Two
independent checks that this matches what the paper used:

- Each split has exactly **414 test-missing ratings**, the number the Logit
  Lens appendix quotes.
- The MCAR mechanism produces exactly 90000/180000 missing — confirming it is
  sampled without replacement, not an i.i.d. coin flip (D6, D15).

The item sweep had to be extended: the paper goes to `K_train = 4000`, the
harness stopped at 1000 (D19).

## 2. Synthetic results

Test missing log-loss, from `best.json` (which is the test-argmin — see D14):

| K_train | Marformer (T) | Marformer NT | | J_train | Marformer (T) | Marformer NT |
|---|---|---|---|---|---|---|
| 100 | 0.9321 | 0.9596 | | 10 | 0.9340 | 0.9155 |
| 200 | 0.8945 | 0.8899 | | 15 | 0.9152 | 0.8922 |
| 300 | 0.8680 | 0.8684 | | 20 | 0.8874 | 0.8770 |
| 400 | 0.8630 | 0.8420 | | 25 | 0.8719 | 0.8357 |
| 500 | 0.8299 | 0.8298 | | 30 | 0.8678 | 0.7923 |
| 600 | 0.8289 | 0.8227 | | 35 | 0.8182 | 0.7985 |
| 700 | 0.8232 | 0.8157 | | 40 | 0.7884 | 0.7913 |
| 800 | 0.8163 | 0.8434 | | 45 | 0.7890 | — |
| 900 | 0.8426 | 0.8346 | | 50 | 0.7768 | — |
| 1000 | 0.8322 | — | | | | |
| 2000 | 0.7843 | — | | | | |
| 3000 | 0.7635 | — | | | | |
| 4000 | **0.7596** | — | | | | |

Both curves decrease monotonically apart from run-to-run wobble, and the item
curve is **still improving at K = 4000** — which is the paper's stated reason
for extending that axis.

### Calibration matches the paper

The paper quotes smECE at two operating points. Recomputed from this run with
the same estimator (`relplot`, 500 bootstrap samples):

| | this run | paper |
|---|---|---|
| Item, K=500, Marformer | **0.014 ± 0.008** | 0.016 ± 0.007 |
| Item, K=500, Marformer NT | **0.012 ± 0.009** | 0.016 ± 0.006 |
| Annot, J=30, Marformer | **0.019 ± 0.009** | 0.022 ± 0.009 |
| Annot, J=30, Marformer NT | **0.018 ± 0.007** | 0.014 ± 0.010 |

All four overlap. The empirical baselines reproduce their ordering too
(Structured Log-Linear 0.215, Structured NB 0.110, Unigram Pool 0.026, Naive
Bayes IJK 0.032 at item K=500), so Marformer being an order of magnitude better
calibrated than the structured baselines replicates.

Stan was not rerun (Tom's call), so the Stan comparison points are absent from
these figures.

## 3. Real data (LLM-Rubric) — replicates closely

| K_train | this run | paper |
|---|---|---|
| 10 | 1.4743 | 1.52 |
| 50 | **1.0494** | 1.050 |
| 175 | **0.9775** | 0.998 |

and the Unigram Pool baseline matches to three decimals (1.1121 vs the paper's
1.112 at K=175; 1.3823 vs 1.38 at K=10).

**This only works if you evaluate at the `max_item` the model trained with.**
`test.py` hardcodes `max_item=500` while these models train at 10, and the error
grows with K — 1.2798 instead of 1.0494 at K=50, and 1.7828 instead of 0.9775 at
K=175. Large enough to invert the trend. The published figures escape it because
the plot script re-scores at `max_item=10`; `best.json` does not, so checkpoint
selection still runs on the wrong numbers. That is D20, and it is worth fixing.

## 4. Figures

All 17 reproducible paper figures regenerate:

```bash
./make_figures.sh --run-id replicate-01 --part all
./make_figures.sh --run-id replicate-01 --cache --install   # into Marformer-Paper/PLOTS
```

12 Domain 3 synthetic + 5 real-data. The 12 orphaned figures (Domains 1 and 2,
SMAR, Logit Lens) have no generating code in this repository — see
[FIGURES.md](../FIGURES.md).

## 5. What this rerun changes about the paper's claims

Nothing about the headline results. But two methodological points surfaced that
should be fixed before submission, both in [DISCREPANCIES.md](../DISCREPANCIES.md):

- **D14** — the synthetic Marformer numbers are a min over 16 checkpoints of
  *test* log-loss. There is no validation split in the synthetic bundles at all,
  so no honest alternative selection exists today. Measured cost of that
  selection: **+0.196 nats at K=100**, +0.028 at K=300, +0.021 at K=500. The
  real-data arm selects on validation and does not have this problem, so the two
  halves of the Domain 3 results are not selected the same way.
- **D20** — `test.py`'s hardcoded `max_item=500` (above).

And one that changes the story rather than the numbers: at small training sizes
the model genuinely overfits, in both arms. See
[OVERFITTING.md](OVERFITTING.md).

## Reproducing

```bash
cd RERUN
./submit_all.sh --run-id myrun --stage data
./submit_all.sh --run-id myrun --stage marformer
./submit_all.sh --run-id myrun --stage eval
./jobs/make_llmrubric_data.sh --run-id myrun
./submit_all.sh --run-id myrun --stage llmrubric
./submit_all.sh --run-id myrun --stage llmrubric-eval
./make_figures.sh --run-id myrun --part all
```

Wall clock on an uncontended cluster: data generation ~15 min, the 38 Marformer
runs 4 min to 3.5 h each (they run concurrently), LLM-Rubric ~1 h, figures
~25 min.
