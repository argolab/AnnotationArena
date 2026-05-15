# Baselines (`imputer/ranking/BASELINES`)

## Structured baselines (categorical)

Missing-cell prediction on `data_bundle.json`: **unigram (ij)**, **NB IJK**, **structured NB**.

→ See **[structured_baselines/README.md](structured_baselines/README.md)** for usage, data format, and model formulas.

```bash
python BASELINES/run_structured_baselines.py --bundle path/to/data_bundle.json
```

## Neural baselines (ReMasker / MIWAE)

Item-matrix imputation with masked training:

```bash
python BASELINES/run_baselines.py \
  --method remasker \
  --data-bundle path/to/data_bundle.json \
  --output-dir RESULTS/BASELINES/run_name
```

See `run_baselines.py --help` for options.

**Calibration plots** (reliability diagrams + smECE; Marformer, STAN, ReMasker, MIWAE, unigram, IJK, SNB):

```bash
python scripts/utils/plot_realdata_calibration.py
python scripts/utils/plot_realdata_calibration.py --dataset LLMRubric --sizes 175
```
