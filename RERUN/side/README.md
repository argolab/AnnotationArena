
## Counting finished runs

`last.ckpt` is **not** a completion marker. Lightning rewrites it every epoch,
so it appears within a minute of a run starting. Counting it reports runs as
finished when they have barely begun -- which is exactly what happened while
monitoring the W3d budget sweep on 2026-09-05 ("4/12 finished" when one run had
finished and three were minutes old).

The completion marker is **`training_history.json`**, which is written once at
the end of training:

    find runs/<tree> -name training_history.json | wc -l

`analyse_arms.py` filters on it, so a partially-trained run cannot silently
enter an arm comparison. `grid_eval.py` and `s7_curves.py` still key off
`last.ckpt` when deciding what to score -- that is safe only when the jobs are
known to have completed, so check job state or the history count before
launching a scoring job.
