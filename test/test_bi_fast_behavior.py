import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_bi_module(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BI


BI_ORIG = _load_bi_module("bi_orig_module", Path("src/prelim/sd/bi_slow.py"))
BI_FAST = _load_bi_module("bi_fast_module", Path("src/prelim/sd/bi.py"))


def _sample(seed, n_rows=160, n_cols=6):
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, 1.0, size=(n_rows, n_cols))
    y = (((X[:, 0] > 0.6) & (X[:, 1] > 0.4)) | (X[:, 2] > 0.8)).astype(int)
    return X, y


def test_bi_fast_matches_original_scores_on_reference_datasets():
    for seed in range(5):
        X, y = _sample(seed)
        original = BI_ORIG(depth=3, beam_size=2, add_iter=5).fit(X, y)
        optimized = BI_FAST(depth=3, beam_size=2, add_iter=5).fit(X, y)

        assert np.isclose(optimized.score(X, y), original.score(X, y))
        assert optimized.get_nrestr() <= optimized.depth


def test_bi_fast_matches_original_when_beam_size_is_one():
    X, y = _sample(123, n_rows=200, n_cols=8)
    original = BI_ORIG(depth=5, beam_size=1, add_iter=10).fit(X, y)
    optimized = BI_FAST(depth=5, beam_size=1, add_iter=10).fit(X, y)

    assert np.allclose(optimized.box_, original.box_)
    assert np.isclose(optimized.score(X, y), original.score(X, y))
