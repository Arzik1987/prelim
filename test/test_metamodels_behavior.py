import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
METAMODELS_DIR = EXPERIMENTS_DIR / "metamodels"


def _load_metamodel_module(module_basename, monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS_DIR))
    module_name = f"metamodels.{module_basename}"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, METAMODELS_DIR / f"{module_basename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_dataset():
    negatives = np.array([[-2.0 + 0.2 * i, -1.0 + 0.1 * i] for i in range(10)])
    positives = np.array([[0.2 + 0.2 * i, 0.1 + 0.1 * i] for i in range(10)])
    X = np.vstack((negatives, positives))
    y = np.array([0] * len(negatives) + [1] * len(positives), dtype=int)
    return X, y


def test_package_exports_legacy_class_names(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS_DIR))
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBClassifier=object))
    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(LGBMClassifier=object))
    import metamodels

    assert "Meta_rf" in metamodels.__all__
    assert "Meta_xgb_bal" in metamodels.__all__
    assert "Meta_lgbm_bal" in metamodels.__all__
    assert callable(metamodels.Meta_svm)


def test_rf_fit_exposes_name_score_and_probability(monkeypatch):
    module = _load_metamodel_module("rf", monkeypatch)
    X, y = _toy_dataset()

    model = module.Meta_rf(params={"max_features": [1]}, cv=2, seed=7).fit(X, y)
    proba = model.predict_proba(X)

    assert model.my_name() == "rf"
    assert model.fit_score() is not None
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_nb_fit_reports_cv_score(monkeypatch):
    module = _load_metamodel_module("nb", monkeypatch)
    X, y = _toy_dataset()

    model = module.Meta_nb().fit(X, y)

    assert model.my_name() == "nb"
    assert model.fit_score() is not None
    assert model.predict(X).shape == (len(X),)


def test_kriging_has_no_cv_score_but_keeps_shared_probability_api(monkeypatch):
    module = _load_metamodel_module("kriging", monkeypatch)
    X, y = _toy_dataset()

    model = module.Meta_kriging(seed=3).fit(X, y)

    assert model.my_name() == "kriging"
    assert model.fit_score() is None
    assert model.predict_proba(X).shape == (len(X),)


def test_svm_default_params_are_initialized(monkeypatch):
    module = _load_metamodel_module("svm", monkeypatch)

    model = module.Meta_svm(cv=2, seed=5)

    assert model.params_["C"] == [0.1, 1, 10, 100]
    assert model.params_["gamma"] == [0.001, 0.01, 0.1, 1]


def test_xgb_variants_keep_names_and_fit_score_contract(monkeypatch):
    class FakeXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.classes_ = np.array([0, 1], dtype=int)

        def fit(self, X, y):
            return self

        def predict(self, X):
            return (X[:, 0] > 0.0).astype(int)

        def predict_proba(self, X):
            pos = np.clip(0.5 + X[:, 0] / 4.0, 0.1, 0.9)
            return np.column_stack((1.0 - pos, pos))

    class FakeRandomizedSearchCV:
        def __init__(self, estimator, params, **kwargs):
            self.estimator = estimator
            self.params = params
            self.kwargs = kwargs
            self.best_estimator_ = estimator
            self.best_score_ = 0.81

        def fit(self, X, y):
            self.best_estimator_.fit(X, y)
            return self

    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBClassifier=FakeXGBClassifier))

    xgb_module = _load_metamodel_module("xgb", monkeypatch)
    xgbb_module = _load_metamodel_module("xgbb", monkeypatch)
    monkeypatch.setattr(xgb_module, "RandomizedSearchCV", FakeRandomizedSearchCV)
    monkeypatch.setattr(xgbb_module, "RandomizedSearchCV", FakeRandomizedSearchCV)

    X, y = _toy_dataset()
    standard = xgb_module.Meta_xgb(cv=2, seed=11).fit(X, y)
    balanced = xgbb_module.Meta_xgb_bal(cv=2, seed=11).fit(X, y)

    assert standard.my_name() == "xgb"
    assert balanced.my_name() == "xgbb"
    assert standard.fit_score() == pytest.approx(0.81)
    assert balanced.fit_score() == pytest.approx(0.81)
    assert balanced.model_.kwargs["scale_pos_weight"] == pytest.approx(1.0)


def test_lgbm_variants_keep_names_and_fit_score_contract(monkeypatch):
    class FakeLGBMClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.classes_ = np.array([0, 1], dtype=int)

        def fit(self, X, y):
            return self

        def predict(self, X):
            return (X[:, 0] > 0.0).astype(int)

        def predict_proba(self, X):
            pos = np.clip(0.5 + X[:, 0] / 4.0, 0.1, 0.9)
            return np.column_stack((1.0 - pos, pos))

    class FakeRandomizedSearchCV:
        def __init__(self, estimator, params, **kwargs):
            self.estimator = estimator
            self.params = params
            self.kwargs = kwargs
            self.best_estimator_ = estimator
            self.best_score_ = 0.77

        def fit(self, X, y):
            self.best_estimator_.fit(X, y)
            return self

    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(LGBMClassifier=FakeLGBMClassifier))

    lgbm_module = _load_metamodel_module("lgbm", monkeypatch)
    lgbmb_module = _load_metamodel_module("lgbmb", monkeypatch)
    monkeypatch.setattr(lgbm_module, "RandomizedSearchCV", FakeRandomizedSearchCV)
    monkeypatch.setattr(lgbmb_module, "RandomizedSearchCV", FakeRandomizedSearchCV)

    X, y = _toy_dataset()
    standard = lgbm_module.Meta_lgbm(cv=2, seed=13).fit(X, y)
    balanced = lgbmb_module.Meta_lgbm_bal(cv=2, seed=13).fit(X, y)

    assert standard.my_name() == "lgbm"
    assert balanced.my_name() == "lgbmb"
    assert standard.fit_score() == pytest.approx(0.77)
    assert balanced.fit_score() == pytest.approx(0.77)
    assert balanced.model_.kwargs["scale_pos_weight"] == pytest.approx(1.0)
