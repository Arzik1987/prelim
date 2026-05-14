import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "experiments" / "data"
EVAL_DIR = ROOT / "experiments" / "evaluation"
RESULTS_DIR = ROOT / "experiments" / "results"


def _load_utils_module(module_basename):
    module_name = f"test_{module_basename}_runner"
    if module_name in sys.modules:
        del sys.modules[module_name]
    module_dir = DATA_DIR if module_basename in {"loader", "partitioner"} else EVAL_DIR
    spec = importlib.util.spec_from_file_location(module_name, module_dir / f"{module_basename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_results_module(module_basename):
    module_name = f"test_results_{module_basename}_runner"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, RESULTS_DIR / f"{module_basename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_data_rejects_unknown_dataset():
    module = _load_utils_module("loader")

    with pytest.raises(ValueError, match="Unknown dataset name"):
        module.load_data("missing-dataset")


def test_load_data_uses_data_dir_and_transforms_occupancy_dates(tmp_path):
    module = _load_utils_module("loader")
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
    module = _load_utils_module("loader")
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


def test_data_partitioner_requires_fit_before_configure():
    module = _load_utils_module("partitioner")
    partitioner = module.DataSplitter(seed=7)

    with pytest.raises(NotFittedError):
        partitioner.configure(2, 1)


def test_data_partitioner_validates_configuration_and_returns_copies():
    module = _load_utils_module("partitioner")
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=int)
    partitioner = module.DataSplitter(seed=11).fit(X, y)

    with pytest.raises(ValueError, match="nparts must be positive"):
        partitioner.configure(0, 2)
    with pytest.raises(ValueError, match="npoints must be positive"):
        partitioner.configure(2, 0)
    with pytest.raises(ValueError, match="at most the fitted sample size"):
        partitioner.configure(2, 11)

    returned = partitioner.configure(3, 2)
    Xtrain, ytrain = partitioner.get_train(1)
    Xtest, ytest = partitioner.get_test(1)

    assert returned is partitioner
    assert Xtrain.shape == (2, 2)
    assert ytrain.shape == (2,)
    assert Xtest.shape == (8, 2)
    assert ytest.shape == (8,)

    original_value = partitioner.X_[partitioner.startpts_[1], 0]
    Xtrain[0, 0] = -999.0
    assert partitioner.X_[partitioner.startpts_[1], 0] == original_value


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


def test_result_writers_flush_after_each_row():
    module = _load_results_module("artifacts")

    class _BufferedHandle:
        def __init__(self):
            self.data = []
            self.flush_count = 0

        def write(self, chunk):
            self.data.append(chunk)
            return len(chunk)

        def flush(self):
            self.flush_count += 1

    result_handle = _BufferedHandle()
    meta_handle = _BufferedHandle()

    module.write_result(result_handle, "dt", "na", "na", 0.1, 0.2, 3, 0.4, "na", 0.5)
    module.write_meta(meta_handle, "overall", 1.23)

    assert result_handle.flush_count == 1
    assert meta_handle.flush_count == 1


def test_build_generators_can_be_scoped_to_forestdiffusion():
    sys.path.insert(0, str(ROOT / "src"))
    from experiments import registries

    generators, genrerx, genvva = registries.build_generators(("forestdiffusion",))

    assert [generator.my_name() for generator in generators] == ["forestdiffusion"]
    assert genrerx.my_name() == "rerx"
    assert genvva.my_name() == "vva"
