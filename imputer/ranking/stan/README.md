# Stan Pipeline

Python pipeline for Stan-based data generation and inference.

## Usage

All commands should be run from the `imputer/ranking/` directory with `PYTHONPATH=.`:

```bash
cd imputer/ranking
source ../../venv-py311/bin/activate
export PYTHONPATH=.
```

### Data Generation

```bash
python stan/scripts/generate_data.py --output-dir runs/data_gen_test --K 10 --I 3 --J 2
```

### Running Tests

```bash
python -m pytest stan/tests/ -v
```

## Structure

- `stan/pipeline/` - Core Python modules
- `stan/scripts/` - CLI interfaces  
- `stan/tests/` - Unit tests
- `models/` - Stan model files

## Dependencies

- Python 3.11+
- cmdstanpy
- numpy
- pytest
