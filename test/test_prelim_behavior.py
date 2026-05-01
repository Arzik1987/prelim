import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from prelim import prelim as prelim_export
from prelim.generators import build_generator
from prelim.prelim import prelim


def _binary_sample():
    rng = np.random.RandomState(2020)
    x0 = rng.normal(loc=-1.0, scale=0.3, size=(30, 2))
    x1 = rng.normal(loc=1.0, scale=0.3, size=(30, 2))
    X = np.vstack((x0, x1))
    y = np.concatenate((np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)))
    return X, y


class _NoProbaMeta:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.majority_ = int(np.argmax(np.bincount(y)))
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_, dtype=int)


class _TrackingMeta(LogisticRegression):
    def __init__(self):
        super().__init__(random_state=2020)
        self.fit_calls_ = 0

    def fit(self, X, y):
        self.fit_calls_ += 1
        return super().fit(X, y)


def test_prelim_rejects_unknown_generator_name():
    X, y = _binary_sample()

    with pytest.raises(ValueError, match="Unknown gen_name"):
        prelim(X, y, LogisticRegression(random_state=2020), DecisionTreeClassifier(), "missing", new_size=90)


def test_public_exports_remain_available():
    assert prelim_export is prelim
    assert build_generator("dummy", seed=2020).my_name() == "dummy"


def test_prelim_requires_predict_proba_when_requested():
    X, y = _binary_sample()

    with pytest.raises(ValueError, match="proba=True requires"):
        prelim(X, y, _NoProbaMeta(), DecisionTreeClassifier(), "dummy", new_size=len(y), proba=True)


def test_prelim_requires_predict_proba_for_vva():
    X, y = _binary_sample()

    with pytest.raises(ValueError, match="Generator 'vva' requires"):
        prelim(X, y, _NoProbaMeta(), DecisionTreeClassifier(), "vva", new_size=len(y))


def test_prelim_rejects_too_small_new_size_for_synthetic_generators():
    X, y = _binary_sample()

    with pytest.raises(ValueError, match="new_size must be at least len\\(y\\)"):
        prelim(X, y, LogisticRegression(random_state=2020), DecisionTreeClassifier(), "norm", new_size=len(y) - 1)


def test_prelim_auto_fits_black_box_model():
    X, y = _binary_sample()
    bb_model = _TrackingMeta()

    model = prelim(X, y, bb_model, DecisionTreeClassifier(random_state=2020), "dummy", new_size=len(y), seed=2020)

    assert bb_model.fit_calls_ == 1
    assert hasattr(model, "tree_")


def test_prelim_vva_path_returns_fitted_white_box_model():
    X, y = _binary_sample()

    model = prelim(
        X,
        y,
        LogisticRegression(random_state=2020),
        DecisionTreeClassifier(random_state=2020, max_depth=3),
        "vva",
        new_size=len(y),
        seed=2020,
    )

    assert hasattr(model, "tree_")
    assert model.score(X, y) >= 0.5
