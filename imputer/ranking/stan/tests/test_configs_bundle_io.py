import json
from pathlib import Path

from stan.pipeline.configs import DataGenConfig, DomainConfig, McmcConfig
from stan.pipeline.io import new_run_dir, save_configs, save_predictives, save_metrics, save_fit_csvs


def test_dataclasses_instantiation():
    dg = DataGenConfig(K=5, I=2, J=2, D=8, C=3)
    dm = DomainConfig(K=5, I=2, J=2, D=8, C=3)
    mc = McmcConfig()
    assert dg.K == 5 and dm.C == 3 and mc.chains >= 1


def test_new_run_dir_and_save_configs(tmp_path):
    run_root = tmp_path / "runs"
    run_dir = new_run_dir(run_root)
    assert run_dir.exists()

    dg = DataGenConfig(K=5, I=2, J=2, D=8, C=3)
    dm = DomainConfig(K=5, I=2, J=2, D=8, C=3)
    mc = McmcConfig()
    save_configs(run_dir, datagen=dg, domain=dm, mcmc=mc)
    cfg_path = run_dir / "configs.json"
    assert cfg_path.exists()
    data = json.loads(cfg_path.read_text())
    assert data["datagen"]["K"] == 5


def test_save_predictives_metrics(tmp_path):
    run_dir = new_run_dir(tmp_path)
    
    save_predictives(run_dir, {"missing_rating_predictions": [2, 1]})
    assert (run_dir / "predictives.json").exists()
    save_metrics(run_dir, {"rating_accuracy": 0.8})
    assert (run_dir / "metrics.json").exists()


class _FakeRunset:
    def __init__(self, csv_files):
        self.csv_files = csv_files


class _FakeFit:
    def __init__(self, csv_files):
        self.runset = _FakeRunset(csv_files)


def test_save_fit_csvs_moves_files(tmp_path):
    run_dir = new_run_dir(tmp_path)
    # Create fake CSV files
    csv_src_dir = tmp_path / "src_csv"
    csv_src_dir.mkdir()
    csv1 = csv_src_dir / "chain1.csv"
    csv2 = csv_src_dir / "chain2.csv"
    csv1.write_text("col1\n1\n")
    csv2.write_text("col1\n2\n")

    fit = _FakeFit([str(csv1), str(csv2)])
    save_fit_csvs(run_dir, fit)

    dest1 = run_dir / "stan_csv" / "chain_1.csv"
    dest2 = run_dir / "stan_csv" / "chain_2.csv"
    assert dest1.exists() and dest2.exists()

