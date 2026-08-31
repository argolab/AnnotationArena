# Domain 3 interactive report

Three pages, same tooling, same splits and seeds:

| Page | Protocol | Marker |
|---|---|---|
| [Domain 3 Replication](https://claude.ai/code/artifact/96812eeb-4c0c-4c9a-a413-67bd321e6282) | none (as the paper runs it) | &mdash; |
| [Domain 3 Validation Protocol](https://claude.ai/code/artifact/5bae9038-9624-4a46-ae8b-d9faa1f8f6b6) | block carve | `BLOCK VAL` |
| [Domain 3 MCAR Validation](https://claude.ai/code/artifact/d5c50d63-a9fd-4d26-9869-f565652e06b3) | MCAR carve | `MCAR VAL` |

The two validation pages carry the first as a dashed overlay, so the difference
on screen is the protocol alone.

Reproduce it from a run tree in three commands:

```bash
python RERUN/analysis/collect_calibration.py --run-dir RERUN/runs/<ID> \
    -o RERUN/runs/<ID>/calibration.json --checkpoints last,best
python RERUN/analysis/collect_domain3.py --run-dir RERUN/runs/<ID> \
    -o /tmp/d3.json --calibration RERUN/runs/<ID>/calibration.json
python RERUN/analysis/build_report.py --data /tmp/d3.json \
    -o RERUN/reports/build/domain3.html [--marker "VAL: BLOCK"]
python RERUN/analysis/check_report.py --page RERUN/reports/build/domain3.html
```

`check_report.py` renders the page headlessly in both themes, drives every
control, and fails on console errors, empty charts, or contrast below 4.5:1.
Run it before publishing; it has caught two real defects already.

## What the page shows

| Section | Content |
|---|---|
| Paper figures | The six generalization sweeps (item / annotator axis, crossed with log-loss, RMSE and training time) as small multiples. The **Arms** control reproduces the paper's `_clean_nt` variants by toggling the non-transductive series. |
| Focus | Any one of those at full size, with a metric picker that also covers accuracy and Spearman. |
| Selected run | Click a point: its test loss at every scored checkpoint, and its training curve, for that exact model. |
| Calibration | The two reliability panels (item K=500, annotator J=30) with bootstrap bands and smECE. |
| Run table | All 114 rows at the selected checkpoint policy. |

## The two controls that carry an argument

**Checkpoint** switches between the last epoch and the paper's policy. There is
no `best-*.ckpt` anywhere in these runs: Lightning's `ModelCheckpoint` monitors
`val/missing_ce`, Domain 3 logs no such metric because it has no validation
split, so the callback never saved anything. The paper's "best" comes entirely
from `plot_domain3_sigmoid._write_best_json_from_test_results`, an argmin over
every scored checkpoint keyed on test log-loss. The page reproduces that rule
exactly, so the toggle measures what test-set selection is worth:

| K (item, transductive) | last | best on test | optimism | selected epoch |
|---|---|---|---|---|
| 100 | 1.0966 | 0.9627 | **+0.134** | 99 |
| 200 | 0.9805 | 0.9142 | +0.066 | 216 |
| 300 | 0.9107 | 0.8840 | +0.027 | 241 |
| 500 | 0.8639 | 0.8521 | +0.012 | 274 |
| 1000 | 0.8862 | 0.8721 | +0.014 | 232 |
| 2000 | 0.7875 | 0.7863 | +0.001 | 332 |
| 4000 | 0.7611 | 0.7610 | +0.000 | 374 |

Mean over seeds 42/43/44. The optimism decays monotonically with training size,
and so does the distance between the selected epoch and the last one -- at
K=4000 the argmin is epoch 374 of 400, i.e. there is nothing to select. This is
discrepancy D14 measured across the whole sweep, and it is what the validation
protocol exists to remove.

**Seeds** switches between the mean with a min-max band and the raw per-seed
lines. Worth using: at K=1000 the three seeds span 0.842 to 0.965, which the
aggregated line hides completely.

## Caveats on the page itself

- Stan curves are absent. The reruns cover Marformer and Marformer NT only,
  per the original scope ("you might not need to run Stan").
- The x axis is **total** items, not training items. `Item_T_K` holds out a
  100-item test block, so training items are K &minus; 100 -- and `Item_T_100`
  has *zero* training items, which is why its combined-eval curve is empty.
- Calibration is reported at two evaluation chunk sizes, `trained` (300, the
  correct one) and `paper` (500 on the item axis, unchunked on the annotator
  axis). That gap is discrepancy D20; on Marformer NT at K=500 it moves smECE
  from 0.043 to 0.025.
