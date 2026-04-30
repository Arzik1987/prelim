import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = ROOT / "experiments" / "utils"


def _load_utils_module(module_basename):
    module_name = f"test_{module_basename}_runner"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, UTILS_DIR / f"{module_basename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_data_rejects_unknown_dataset():
    module = _load_utils_module("data_loader")

    with pytest.raises(ValueError, match="Unknown dataset name"):
        module.load_data("missing-dataset")


def test_load_data_uses_data_dir_and_transforms_occupancy_dates(tmp_path):
    module = _load_utils_module("data_loader")
    occupancy_dir = tmp_path / "occupancy"
    occupancy_dir.mkdir(parents=True)

    frame_a = pd.DataFrame(
        {
            "date": ["2024-01-01 03:15:00", "2024-01-01 04:00:00"],
            "Temperature": [21.5, 22.0],
            "Occupancy": [1, 0],
        }
    )
    frame_b = pd.DataFrame(
        {
            "date": ["2024-01-01 05:00:00"],
            "Temperature": [23.0],
            "Occupancy": [1],
        }
    )

    frame_a.to_csv(occupancy_dir / "datatest.txt", index=False)
    frame_b.to_csv(occupancy_dir / "datatest2.txt", index=False)
    frame_b.to_csv(occupancy_dir / "datatraining.txt", index=False)

    X, y = module.load_data("occupancy", data_dir=tmp_path)

    assert X.shape == (4, 2)
    assert y.tolist() == [1, 0, 1, 1]
    assert X[:, 0].tolist() == [3, 4, 5, 5]


def test_load_data_converts_jm1_missing_markers_and_drops_nan_rows(tmp_path):
    module = _load_utils_module("data_loader")
    jm1_dir = tmp_path / "jm1"
    jm1_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "f1": ["1.5", "?"],
            "f2": ["2.5", "3.5"],
            "defects": [True, False],
        }
    ).to_csv(jm1_dir / "jm1.csv", index=False)

    X, y = module.load_data("jm1", data_dir=tmp_path)

    assert X.dtype == np.float64
    assert X.tolist() == [[1.5, 2.5]]
    assert y.tolist() == [1]


def test_data_splitter_requires_fit_before_configure():
    module = _load_utils_module("data_splitter")
    splitter = module.DataSplitter(seed=7)

    with pytest.raises(NotFittedError):
        splitter.configure(2, 1)


def test_data_splitter_validates_configuration_and_returns_copies():
    module = _load_utils_module("data_splitter")
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=int)
    splitter = module.DataSplitter(seed=11).fit(X, y)

    with pytest.raises(ValueError, match="nparts must be positive"):
        splitter.configure(0, 2)
    with pytest.raises(ValueError, match="npoints must be positive"):
        splitter.configure(2, 0)
    with pytest.raises(ValueError, match="at most the fitted sample size"):
        splitter.configure(2, 11)

    returned = splitter.configure(3, 2)
    Xtrain, ytrain = splitter.get_train(1)
    Xtest, ytest = splitter.get_test(1)

    assert returned is splitter
    assert Xtrain.shape == (2, 2)
    assert ytrain.shape == (2,)
    assert Xtest.shape == (8, 2)
    assert ytest.shape == (8,)

    original_value = splitter.X_[splitter.startpts_[1], 0]
    Xtrain[0, 0] = -999.0
    assert splitter.X_[splitter.startpts_[1], 0] == original_value


def test_opt_param_averages_only_split_scores():
    module = _load_utils_module("helpers")
    cvres = {
        "split0_test_score": np.array([0.1, 0.6, np.nan]),
        "split1_test_score": np.array([0.3, 0.4, 0.9]),
        "mean_test_score": np.array([0.2, 0.5, 0.9]),
    }

    result = module.opt_param(cvres, 3)

    assert np.allclose(result[:2], np.array([0.2, 0.5]))
    assert result[2] == 0.9


def test_n_leaves_counts_internal_splits_for_simple_tree():
    module = _load_utils_module("helpers")
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=1, random_state=2020).fit(X, y)

    assert module.n_leaves(tree) == 2


def test_get_bi_param_caps_attribute_count():
    module = _load_utils_module("helpers")

    result = module.get_bi_param(5, 20)

    assert result.tolist() == [3, 6, 9, 12, 15]


def test_get_new_test_never_uses_negative_pool_size():
    module = _load_utils_module("helpers")
    Xtest = np.arange(12, dtype=float).reshape(6, 2)
    ytest = np.array([0, 1, 0, 1, 0, 1])

    kept_X, kept_y, new_X = module.get_new_test(Xtest, ytest, dsize=8, new_size=6)

    assert np.array_equal(kept_X, Xtest)
    assert np.array_equal(kept_y, ytest)
    assert new_X.shape == (0, 2)
