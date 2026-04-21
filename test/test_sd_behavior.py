import numpy as np
import pytest

from prelim.sd.bi import BI
from prelim.sd.prim import PRIM


def _sd_sample():
    rng = np.random.RandomState(2020)
    X = rng.uniform(0.0, 1.0, size=(80, 3))
    y = ((X[:, 0] > 0.55) & (X[:, 1] > 0.45)).astype(int)
    return X, y


def test_bi_fit_populates_box_and_score_is_finite():
    X, y = _sd_sample()
    model = BI(depth=3, beam_size=2, add_iter=5).fit(X, y)

    score = model.score(X, y)

    assert model.box_.shape == (2, X.shape[1])
    assert np.isfinite(score)
    assert model.get_nrestr() >= 0


def test_bi_warns_when_depth_exceeds_dimensions():
    X, y = _sd_sample()

    with pytest.warns(UserWarning, match="Restricting depth parameter"):
        BI(depth=10).fit(X, y)


def test_bi_params_round_trip():
    model = BI(depth=2, beam_size=3, add_iter=7)

    assert model.get_params() == {"beam_size": 3, "depth": 2, "add_iter": 7}
    assert model.set_params(depth=4, beam_size=1) is model
    assert model.depth == 4
    assert model.beam_size == 1


def test_prim_fit_populates_box_and_score_is_finite():
    X, y = _sd_sample()
    model = PRIM(alpha=0.1).fit(X, y)

    score = model.score(X, y)

    assert model.box_.shape == (2, X.shape[1])
    assert np.isfinite(score)
    assert model.get_nrestr() >= 0


def test_prim_params_round_trip():
    model = PRIM(alpha=0.2)

    assert model.get_params() == {"alpha": 0.2}
    assert model.set_params(alpha=0.15) is model
    assert model.alpha == 0.15


def test_prim_score_returns_nan_when_box_selects_no_rows():
    X, y = _sd_sample()
    model = PRIM(alpha=0.1).fit(X, y)
    model.box_ = np.array([[10.0, -np.inf, -np.inf], [11.0, np.inf, np.inf]])

    score = model.score(X, y)

    assert np.isnan(score)
