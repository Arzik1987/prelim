import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path
import csv

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def _repeat_rows(X, n_rows):
    reps = int(np.ceil(n_rows / len(X)))
    return np.tile(X, (reps, 1))[:n_rows].copy()


class _StubRuleModel:
    def __init__(self, max_rules=8, **kwargs):
        self.max_rules = max_rules
        self.ruleset_ = []

    def fit(self, X, y):
        self.ruleset_ = [("x0>0", self.max_rules)]
        return self

    def predict(self, X):
        return (X[:, 0] > 0.0).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))


class _StubMetaModel:
    def __init__(self, name):
        self._name = name
        self.threshold_ = 0.0
        self.cvscore_ = 0.75

    def fit(self, X, y):
        self.threshold_ = float(np.median(X[:, 0]))
        return self

    def predict(self, X):
        return (X[:, 0] > self.threshold_).astype(int)

    def predict_proba(self, X):
        return np.clip(0.5 + (X[:, 0] - self.threshold_) / 2.0, 0.05, 0.95)

    def fit_score(self):
        return self.cvscore_

    def my_name(self):
        return self._name


class _StubGenerator:
    def __init__(self, name):
        self._name = name
        self.X_ = None

    def fit(self, X, *args):
        self.X_ = X.copy()
        return self

    def sample(self, n_samples=10):
        return _repeat_rows(self.X_, n_samples)

    def my_name(self):
        return self._name


class _StubRerx(_StubGenerator):
    def fit(self, X, y, meta_model):
        self.X_ = X.copy()
        return self

    def sample(self):
        return _repeat_rows(self.X_, len(self.X_))


class _StubVva(_StubGenerator):
    def fit(self, X, meta_model):
        self.X_ = X.copy()
        return self

    def will_generate(self):
        return True

    def sample(self, r=1.0):
        n_rows = max(1, int(round(len(self.X_) * r)))
        return _repeat_rows(self.X_, n_rows)


def _install_experiment_stubs(monkeypatch):
    fake_wittgenstein = types.SimpleNamespace(RIPPER=_StubRuleModel, IREP=_StubRuleModel)
    fake_xgboost = types.SimpleNamespace(XGBClassifier=object)
    fake_lightgbm = types.SimpleNamespace(LGBMClassifier=object)
    fake_imblearn_over_sampling = types.SimpleNamespace(ADASYN=object, SMOTE=object)
    fake_imblearn = types.SimpleNamespace(over_sampling=fake_imblearn_over_sampling)
    monkeypatch.setitem(sys.modules, "wittgenstein", fake_wittgenstein)
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lightgbm)
    monkeypatch.setitem(sys.modules, "imblearn", fake_imblearn)
    monkeypatch.setitem(sys.modules, "imblearn.over_sampling", fake_imblearn_over_sampling)
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    monkeypatch.syspath_prepend(str(EXPERIMENTS_DIR))


def _load_experiments_module(monkeypatch):
    _install_experiment_stubs(monkeypatch)
    module_name = "experiments_smoke_runner"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, EXPERIMENTS_DIR / "experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_dataset():
    x0 = np.array(
        [
            [-2.0, -1.0],
            [-1.8, -0.9],
            [-1.6, -0.8],
            [-1.4, -0.7],
            [-1.2, -0.6],
            [-1.0, -0.5],
            [-0.8, -0.4],
            [-0.6, -0.3],
            [-0.4, -0.2],
            [-0.2, -0.1],
        ]
    )
    x1 = np.array(
        [
            [0.2, 0.1],
            [0.4, 0.2],
            [0.6, 0.3],
            [0.8, 0.4],
            [1.0, 0.5],
            [1.2, 0.6],
            [1.4, 0.7],
            [1.6, 0.8],
            [1.8, 0.9],
            [2.0, 1.0],
        ]
    )
    X = np.vstack((x0, x1))
    y = np.array([0] * len(x0) + [1] * len(x1), dtype=int)
    return X, y


def _patch_smoke_components(monkeypatch, module):
    monkeypatch.setattr(module, "load_data", lambda dataset_name: _tiny_dataset())
    monkeypatch.setattr(module, "build_generators", lambda generator_names=None: ([_StubGenerator("stubgen")], _StubRerx("rerx"), _StubVva("vva")))
    monkeypatch.setattr(module._run, "build_generators", lambda generator_names=None: ([_StubGenerator("stubgen")], _StubRerx("rerx"), _StubVva("vva")))
    monkeypatch.setattr(module, "Gen_rerx", lambda: _StubRerx("rerx"))
    monkeypatch.setattr(module, "Gen_vva", lambda: _StubVva("vva"))
    monkeypatch.setattr(
        module,
        "STANDARD_METAMODEL_FACTORIES",
        (
            lambda: _StubMetaModel("rf"),
        ),
    )
    monkeypatch.setattr(
        module,
        "BALANCED_METAMODEL_FACTORIES",
        (
            lambda: _StubMetaModel("rfb"),
        ),
    )
    monkeypatch.setattr(module, "is_balanced_metamodel", lambda model: model.my_name().endswith("b"))


def _read_raw_rows(raw_path):
    with raw_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_build_config_accepts_metamodel_cli_selection(monkeypatch):
    module = _load_experiments_module(monkeypatch)
    args = types.SimpleNamespace(
        run_id="configured-run",
        datasets="ccpp",
        sizes="100",
        nsets=1,
        split_seed=2020,
        jobs=1,
        generated_sample_size=100,
        rules_sample_size=50,
        ssl_pool_size=50,
        vva_grid="0.5,1.0",
        generators="dummy",
        standard_metamodels="rf",
        balanced_metamodels="rf",
        skip_rerx=False,
        skip_vva=False,
        skip_ssl=False,
        resume=False,
    )

    config = module.build_config(args)

    assert config.standard_metamodels == ("rf",)
    assert config.balanced_metamodels == ("rf",)


def test_build_config_accepts_empty_white_box_model_groups(monkeypatch):
    module = _load_experiments_module(monkeypatch)
    args = types.SimpleNamespace(
        run_id="configured-run",
        datasets="ccpp",
        sizes="100",
        nsets=1,
        split_seed=2020,
        jobs=1,
        generated_sample_size=100,
        rules_sample_size=50,
        ssl_pool_size=50,
        vva_grid="0.5,1.0",
        generators="dummy",
        standard_metamodels="rf",
        balanced_metamodels="rf",
        tree_models="",
        balanced_tree_models="",
        rule_models="",
        sd_models="",
        skip_rerx=False,
        skip_vva=False,
        skip_ssl=False,
        resume=False,
    )

    config = module.build_config(args)

    assert config.tree_models == ()
    assert config.balanced_tree_models == ()
    assert config.rule_models == ()
    assert config.sd_models == ()


def test_build_config_accepts_empty_metamodel_groups(monkeypatch):
    module = _load_experiments_module(monkeypatch)
    args = types.SimpleNamespace(
        run_id="configured-run",
        datasets="ccpp",
        sizes="100",
        nsets=1,
        split_seed=2020,
        jobs=1,
        generated_sample_size=100,
        rules_sample_size=50,
        ssl_pool_size=50,
        vva_grid="0.5,1.0",
        generators="dummy",
        standard_metamodels="",
        balanced_metamodels="",
        tree_models="dtc",
        balanced_tree_models="dtcb",
        rule_models="grl",
        sd_models="bicv",
        skip_rerx=False,
        skip_vva=False,
        skip_ssl=False,
        resume=False,
    )

    config = module.build_config(args)

    assert config.standard_metamodels == ()
    assert config.balanced_metamodels == ()


def test_build_config_rejects_unknown_metamodel_name(monkeypatch):
    module = _load_experiments_module(monkeypatch)
    args = types.SimpleNamespace(
        run_id="configured-run",
        datasets="ccpp",
        sizes="100",
        nsets=1,
        split_seed=2020,
        jobs=1,
        generated_sample_size=100,
        rules_sample_size=50,
        ssl_pool_size=50,
        vva_grid="0.5,1.0",
        generators="dummy",
        standard_metamodels="missing",
        balanced_metamodels="rf",
        skip_rerx=False,
        skip_vva=False,
        skip_ssl=False,
        resume=False,
    )

    with pytest.raises(ValueError, match="Unknown standard metamodel"):
        module.build_config(args)


def test_build_model_groups_accept_empty_white_box_model_groups(monkeypatch, tmp_path):
    module = _load_experiments_module(monkeypatch)
    config = module.ExperimentConfig(
        run_id="empty-wb-selection",
        datasets=("toy",),
        dataset_sizes=(10,),
        standard_metamodels=("rf",),
        balanced_metamodels=("rf",),
        tree_models=(),
        balanced_tree_models=(),
        rule_models=(),
        sd_models=(),
        registry_dir=str(tmp_path / "registry"),
    )

    assert module.build_tree_models(config) == {}
    assert module.build_balanced_tree_models(config) == {}
    assert module.build_rule_models(config) == {}


def test_build_metamodel_groups_accept_empty_groups(monkeypatch, tmp_path):
    module = _load_experiments_module(monkeypatch)
    config = module.ExperimentConfig(
        run_id="empty-meta-selection",
        datasets=("toy",),
        dataset_sizes=(10,),
        standard_metamodels=(),
        balanced_metamodels=(),
        registry_dir=str(tmp_path / "registry"),
    )

    standard, balanced = module.build_metamodel_groups(config)

    assert standard == []
    assert balanced == []


def test_build_metamodel_groups_uses_configured_metamodel_names(monkeypatch, tmp_path):
    module = _load_experiments_module(monkeypatch)
    config = module.ExperimentConfig(
        run_id="metamodel-selection",
        datasets=("toy",),
        dataset_sizes=(10,),
        standard_metamodels=("rf",),
        balanced_metamodels=("rf",),
        registry_dir=str(tmp_path / "registry"),
    )

    standard, balanced = module.build_metamodel_groups(config)

    assert [model.my_name() for model in standard] == ["rf"]
    assert [model.my_name() for model in balanced] == ["rfb"]


def test_exp_parallel_smoke_creates_versioned_run_artifacts(monkeypatch, tmp_path):
    module = _load_experiments_module(monkeypatch)
    _patch_smoke_components(monkeypatch, module)

    config = module.ExperimentConfig(
        run_id="smoke-run",
        datasets=("toy",),
        dataset_sizes=(10,),
        nsets=1,
        split_seed=2020,
        generated_sample_size=20,
        rules_sample_size=10,
        ssl_pool_size=10,
        vva_grid=(0.5, 1.0),
        jobs=1,
        registry_dir=str(tmp_path / "registry"),
    )

    module.ensure_run_layout(config)
    module.configure_logging(config)
    module.write_manifest(config, status="running")
    result_list, summary = module.exp_parallel(config)
    module.write_manifest(config, status="completed", summary=summary)

    assert result_list[0][0] == "completed"
    assert summary == {"completed": 1, "failed": 0, "skipped": 0, "total": 1, "zero_class": 0}

    raw_path = Path(module.result_paths(config, "toy", 0, 10)["raw"])
    meta_path = Path(module.result_paths(config, "toy", 0, 10)["meta"])
    manifest_path = Path(config.manifest_path)

    assert raw_path.exists()
    assert meta_path.exists()
    assert manifest_path.exists()
    raw_rows = _read_raw_rows(raw_path)
    alg_gen_pairs = {(row[0], row[1]) for row in raw_rows}
    assert ("dt", "na") in alg_gen_pairs
    assert ("dt", "stubgen") in alg_gen_pairs
    assert ("dtp", "stubgen") not in alg_gen_pairs
    assert ("dt", "ssl") in alg_gen_pairs
    assert ("dt", "ssl_oracle") in alg_gen_pairs
    assert "stubgentime" in meta_path.read_text(encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "smoke-run"
    assert manifest["status"] == "completed"
    assert manifest["summary"]["completed"] == 1


def test_resume_mode_skips_completed_experiment_shards(monkeypatch, tmp_path):
    module = _load_experiments_module(monkeypatch)
    _patch_smoke_components(monkeypatch, module)

    base_config = module.ExperimentConfig(
        run_id="resume-run",
        datasets=("toy",),
        dataset_sizes=(10,),
        nsets=1,
        split_seed=2020,
        generated_sample_size=20,
        rules_sample_size=10,
        ssl_pool_size=10,
        vva_grid=(1.0,),
        jobs=1,
        registry_dir=str(tmp_path / "registry"),
    )

    module.ensure_run_layout(base_config)
    module.configure_logging(base_config)
    first_results, first_summary = module.exp_parallel(base_config)
    assert first_results[0][0] == "completed"
    assert first_summary["completed"] == 1

    resume_config = replace(base_config, resume=True)
    module.ensure_run_layout(resume_config)
    resumed_results, resumed_summary = module.exp_parallel(resume_config)

    assert resumed_results[0][0] == "skipped"
    assert resumed_summary == {"completed": 0, "failed": 0, "skipped": 1, "total": 1, "zero_class": 0}


def test_generated_only_tree_models_can_be_reenabled(monkeypatch, tmp_path):
    module = _load_experiments_module(monkeypatch)
    _patch_smoke_components(monkeypatch, module)

    config = module.ExperimentConfig(
        run_id="generated-only-run",
        datasets=("toy",),
        dataset_sizes=(10,),
        nsets=1,
        split_seed=2020,
        generated_sample_size=20,
        rules_sample_size=10,
        ssl_pool_size=10,
        vva_grid=(0.5, 1.0),
        include_generated_only_tree_models=True,
        jobs=1,
        registry_dir=str(tmp_path / "registry"),
    )

    module.ensure_run_layout(config)
    module.configure_logging(config)
    result_list, summary = module.exp_parallel(config)

    assert result_list[0][0] == "completed"
    assert summary["completed"] == 1

    raw_path = Path(module.result_paths(config, "toy", 0, 10)["raw"])
    alg_gen_pairs = {(row[0], row[1]) for row in _read_raw_rows(raw_path)}

    assert ("dtp", "stubgen") in alg_gen_pairs
    assert ("dtcp", "stubgen") in alg_gen_pairs
    assert ("dtvalp", "stubgen") in alg_gen_pairs


def test_skip_auxiliary_phases_omit_rerx_vva_and_ssl_outputs(monkeypatch, tmp_path):
    module = _load_experiments_module(monkeypatch)
    _patch_smoke_components(monkeypatch, module)

    config = module.ExperimentConfig(
        run_id="skip-aux-phases",
        datasets=("toy",),
        dataset_sizes=(10,),
        nsets=1,
        split_seed=2020,
        generated_sample_size=20,
        rules_sample_size=10,
        ssl_pool_size=10,
        vva_grid=(0.5, 1.0),
        skip_rerx=True,
        skip_vva=True,
        skip_ssl=True,
        jobs=1,
        registry_dir=str(tmp_path / "registry"),
    )

    module.ensure_run_layout(config)
    module.configure_logging(config)
    result_list, summary = module.exp_parallel(config)

    assert result_list[0][0] == "completed"
    assert summary["completed"] == 1

    raw_path = Path(module.result_paths(config, "toy", 0, 10)["raw"])
    alg_gen_pairs = {(row[0], row[1]) for row in _read_raw_rows(raw_path)}

    assert ("dt", "stubgen") in alg_gen_pairs
    assert ("dt", "rerx") not in alg_gen_pairs
    assert ("dt", "vva") not in alg_gen_pairs
    assert ("dt", "ssl") not in alg_gen_pairs
    assert ("dt", "ssl_oracle") not in alg_gen_pairs
