import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def _repeat_rows(X, n_rows):
    reps = int(np.ceil(n_rows / len(X)))
    return np.tile(X, (reps, 1))[:n_rows].copy()


class _StubRuleModel:
    def __init__(self, max_rules=8):
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
    monkeypatch.setitem(sys.modules, "wittgenstein", fake_wittgenstein)
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgboost)
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lightgbm)
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
    monkeypatch.setattr(
        module,
        "GENERATOR_FACTORIES",
        (
            lambda: _StubGenerator("stubgen"),
        ),
    )
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
    assert "dt,na,na" in raw_path.read_text(encoding="utf-8")
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
