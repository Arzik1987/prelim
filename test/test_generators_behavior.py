from pathlib import Path

import numpy as np
import pytest

from prelim.generators import Gen_dummy as Gen_dummy_export
from prelim.generators import Gen_bayesnet
from prelim.generators import Gen_binarydiffusion
from prelim.generators import Gen_copulagan
from prelim.generators import Gen_ctgan
from prelim.generators import Gen_forestdiffusion
from prelim.generators import Gen_gaussiancopula
from prelim.generators import Gen_great
from prelim.generators import Gen_tabgan
from prelim.generators import Gen_tabddpm
from prelim.generators import Gen_tabsyn
from prelim.generators import Gen_tvae
from prelim.generators import Gen_vva_proba as Gen_vva_proba_export
from prelim.generators import build_generator
from prelim.generators.adasyn import Gen_adasyn
from prelim.generators.dummy import Gen_dummy
from prelim.generators.gmm import Gen_gmm, Gen_gmmbic, Gen_gmmbical
from prelim.generators.kde import Gen_kdebw, Gen_kdebwhl
from prelim.generators.kdeb import Gen_kdeb
from prelim.generators.kdem import Gen_kdebwm
from prelim.generators.munge import Gen_munge
from prelim.generators.noise import Gen_noise
from prelim.generators.perfect import Gen_perfect
from prelim.generators.rand import Gen_randn, Gen_randu
from prelim.generators.rerx import Gen_rerx
from prelim.generators.rfdens import Gen_rfdens
from prelim.generators.smote import Gen_smote
from prelim.generators.vva import Gen_vva as Gen_vva_legacy
from prelim.generators.vva_p import Gen_vva as Gen_vva_proba


def _clustered_sample():
    rng = np.random.RandomState(2020)
    x1 = rng.multivariate_normal([0.0, 0.0], np.eye(2) * 0.2, 40)
    x2 = rng.multivariate_normal([3.0, 3.0], np.eye(2) * 0.2, 40)
    return np.vstack((x1, x2))


def _grid_sample():
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ]
    )


def _labeled_clustered_sample():
    x = _clustered_sample()
    y = np.concatenate((np.zeros(40, dtype=int), np.ones(40, dtype=int)))
    return x, y


class _AllOnesMeta:
    def predict(self, x):
        return np.ones(len(x), dtype=int)


class _LinearProbabilityMeta:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p1 = np.clip(0.5 + x[:, 0] / 6.0, 0.01, 0.99)
        return np.column_stack((1.0 - p1, p1))


class _LinearScoreMeta:
    def predict_proba(self, x):
        return np.clip(0.5 + x[:, 0] / 6.0, 0.01, 0.99)


class _SingleSideProbabilityMeta:
    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p1 = np.full(len(x), 0.9)
        return np.column_stack((1.0 - p1, p1))


class _SingleSideScoreMeta:
    def predict_proba(self, x):
        return np.full(len(x), 0.9)


def test_dummy_returns_full_copy_of_fitted_data():
    x = _clustered_sample()
    generator = Gen_dummy().fit(x)

    sample = generator.sample(n_samples=5)

    assert np.array_equal(sample, x)
    assert sample is not x
    sample[0, 0] = -999.0
    assert generator.X_[0, 0] != -999.0
    assert generator.my_name() == "dummy"


def test_generator_package_exports_public_surface():
    assert Gen_dummy_export is Gen_dummy
    assert Gen_vva_proba_export is Gen_vva_proba
    assert build_generator("dummy", seed=2020).my_name() == "dummy"


def test_tabgan_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeGANGenerator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate_data_pipe(self, train_df, target, test_df, **kwargs):
            rows = train_df.iloc[:3, :].copy().reset_index(drop=True)
            rows.iloc[:, :] = np.arange(rows.size).reshape(rows.shape)
            return rows, None

    monkeypatch.setattr("prelim.generators.tabgan.GANGenerator", _FakeGANGenerator)

    x = _clustered_sample()
    generator = Gen_tabgan(generator_kwargs={"gen_x_times": 1.1}, seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "tabgan"
    assert build_generator("tabgan", seed=2020).my_name() == "tabgan"


def test_ctgan_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeCTGAN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_shape_ = None

        def fit(self, data, discrete_columns):
            self.fit_shape_ = data.shape
            self.discrete_columns_ = discrete_columns

        def sample(self, n_samples):
            return __import__("pandas").DataFrame(np.arange(n_samples * 2).reshape(n_samples, 2))

    monkeypatch.setattr("prelim.generators.ctgan.CTGAN", _FakeCTGAN)

    x = _clustered_sample()
    generator = Gen_ctgan(model_kwargs={"epochs": 1}, seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "ctgan"
    assert generator.model_.fit_shape_ == x.shape
    assert generator.model_.discrete_columns_ == []
    assert build_generator("ctgan", seed=2020).my_name() == "ctgan"


def test_tvae_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeTVAE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_shape_ = None

        def fit(self, data, discrete_columns):
            self.fit_shape_ = data.shape
            self.discrete_columns_ = discrete_columns

        def sample(self, n_samples):
            return __import__("pandas").DataFrame(np.arange(n_samples * 2).reshape(n_samples, 2))

    monkeypatch.setattr("prelim.generators.tvae.TVAE", _FakeTVAE)

    x = _clustered_sample()
    generator = Gen_tvae(model_kwargs={"epochs": 1}, seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "tvae"
    assert generator.model_.fit_shape_ == x.shape
    assert generator.model_.discrete_columns_ == []
    assert build_generator("tvae", seed=2020).my_name() == "tvae"


def test_copulagan_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeMetadata:
        def __init__(self):
            self.detected_shape_ = None

        def detect_from_dataframe(self, data):
            self.detected_shape_ = data.shape

    class _FakeCopulaGAN:
        def __init__(self, metadata, **kwargs):
            self.metadata = metadata
            self.kwargs = kwargs
            self.fit_shape_ = None

        def fit(self, data):
            self.fit_shape_ = data.shape

        def sample(self, num_rows):
            return __import__("pandas").DataFrame(np.arange(num_rows * 2).reshape(num_rows, 2))

    monkeypatch.setattr("prelim.generators.copulagan.SingleTableMetadata", _FakeMetadata)
    monkeypatch.setattr("prelim.generators.copulagan.CopulaGANSynthesizer", _FakeCopulaGAN)

    x = _clustered_sample()
    generator = Gen_copulagan(model_kwargs={"epochs": 1}, seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "copulagan"
    assert generator.metadata_.detected_shape_ == x.shape
    assert generator.model_.fit_shape_ == x.shape
    assert build_generator("copulagan", seed=2020).my_name() == "copulagan"


def test_gaussiancopula_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeMetadata:
        def __init__(self):
            self.detected_shape_ = None

        def detect_from_dataframe(self, data):
            self.detected_shape_ = data.shape

    class _FakeGaussianCopula:
        def __init__(self, metadata, **kwargs):
            self.metadata = metadata
            self.kwargs = kwargs
            self.fit_shape_ = None

        def fit(self, data):
            self.fit_shape_ = data.shape

        def sample(self, num_rows):
            return __import__("pandas").DataFrame(np.arange(num_rows * 2).reshape(num_rows, 2))

    monkeypatch.setattr("prelim.generators.gaussiancopula.SingleTableMetadata", _FakeMetadata)
    monkeypatch.setattr("prelim.generators.gaussiancopula.GaussianCopulaSynthesizer", _FakeGaussianCopula)

    x = _clustered_sample()
    generator = Gen_gaussiancopula(seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "gaussiancopula"
    assert generator.metadata_.detected_shape_ == x.shape
    assert generator.model_.fit_shape_ == x.shape
    assert build_generator("gaussiancopula", seed=2020).my_name() == "gaussiancopula"


def test_great_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeGReaT:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_shape_ = None

        def fit(self, data):
            self.fit_shape_ = data.shape

        def sample(self, n_samples):
            return __import__("pandas").DataFrame(np.arange(n_samples * 2).reshape(n_samples, 2))

    monkeypatch.setattr("prelim.generators.great.GReaT", _FakeGReaT)

    x = _clustered_sample()
    generator = Gen_great(model_kwargs={"epochs": 1}, seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "great"
    assert generator.model_.fit_shape_ == x.shape
    assert build_generator("great", seed=2020).my_name() == "great"


def test_tabddpm_generator_runs_repo_actions_and_returns_requested_shape(monkeypatch, tmp_path):
    repo_root = tmp_path / "tabddpm"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "train.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "sample.py").write_text("", encoding="utf-8")

    calls = []

    def _fake_run_action(self, action, kwargs):
        calls.append((action, kwargs))
        output_root = Path(kwargs["parent_dir"])
        if action == "train":
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "model.pt").write_text("ok", encoding="utf-8")
        else:
            np.save(output_root / "X_num_train.npy", np.arange(kwargs["num_samples"] * 2).reshape(kwargs["num_samples"], 2))

    monkeypatch.setattr("prelim.generators.tabddpm.Gen_tabddpm._run_action", _fake_run_action)

    x = _clustered_sample()
    generator = Gen_tabddpm(repo_path=repo_root, seed=2020).fit(x)
    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "tabddpm"
    assert build_generator("tabddpm", seed=2020).my_name() == "tabddpm"
    assert [action for action, _ in calls] == ["train", "sample"]
    assert calls[0][1]["model_params"]["num_classes"] == 2


def test_tabddpm_generator_writes_helper_target_and_preserves_column_order(tmp_path):
    repo_root = tmp_path / "tabddpm"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "train.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "sample.py").write_text("", encoding="utf-8")

    x = np.array(
        [
            [0.0, "a", 10.0],
            [1.0, "b", 20.0],
            [2.0, "a", 30.0],
            [3.0, "b", 40.0],
        ],
        dtype=object,
    )
    generator = Gen_tabddpm(repo_path=repo_root, keep_artifacts=True, seed=2020)
    generator.repo_root_ = repo_root
    generator.run_root_ = tmp_path / "run"
    generator.data_root_ = generator.run_root_ / "data"
    generator.output_root_ = generator.run_root_ / "output"
    generator.data_root_.mkdir(parents=True)
    generator.output_root_.mkdir(parents=True)
    generator.X_ = x
    generator.column_specs_ = generator._analyze_columns(x)
    generator._write_dataset_artifacts()

    y_train = np.load(generator.data_root_ / "y_train.npy", allow_pickle=True)
    x_num_train = np.load(generator.data_root_ / "X_num_train.npy", allow_pickle=True)
    x_cat_train = np.load(generator.data_root_ / "X_cat_train.npy", allow_pickle=True)
    np.save(generator.output_root_ / "X_num_train.npy", x_num_train)
    np.save(generator.output_root_ / "X_cat_train.npy", x_cat_train)

    rebuilt = generator._load_sampled_rows()

    assert set(np.unique(y_train)).issubset({0, 1})
    assert rebuilt.shape[1] == x.shape[1]
    assert rebuilt[0, 1] == "a"
    assert float(rebuilt[1, 2]) == 20.0


def test_binarydiffusion_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeDataset:
        def __init__(self, table, target_column, split_feature_target, task, numerical_columns, categorical_columns):
            self.table = table
            self.target_column = target_column
            self.split_feature_target = split_feature_target
            self.task = task
            self.numerical_columns = numerical_columns
            self.categorical_columns = categorical_columns
            self.row_size = table.shape[1] * 4
            self.n_classes = 2
            self.conditional = True
            self.transformation = self

        def inverse_transform(self, rows, labels):
            arr = rows.detach().cpu().numpy()
            frame = __import__("pandas").DataFrame(arr[:, :2], columns=["x0", "x1"])
            return frame, labels.detach().cpu().numpy()

    class _FakeModel:
        def __init__(self, **kwargs):
            self.config = kwargs
            self.data_dim = kwargs["data_dim"]
            self.out_dim = kwargs["out_dim"]
            self.conditional = kwargs["conditional"]
            self.n_classes = kwargs["n_classes"]
            self.classifier_free_guidance = kwargs["classifier_free_guidance"]

        def to(self, device):
            self.device = device
            return self

    class _FakeDiffusion:
        def __init__(self, denoise_model, **kwargs):
            self.model = denoise_model
            self.conditional = denoise_model.conditional
            self.n_classes = denoise_model.n_classes
            self.classifier_free_guidance = denoise_model.classifier_free_guidance
            self.kwargs = kwargs

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def sample(self, **kwargs):
            n = kwargs["n"]
            return __import__("torch").tensor(np.arange(n * 8).reshape(n, 8)).float()

    class _FakeEMA:
        def __init__(self, diffusion):
            self.ema_model = diffusion

    class _FakeTrainer:
        def __init__(self, diffusion, dataset, **kwargs):
            self.diffusion = diffusion
            self.dataset = dataset
            self.kwargs = kwargs
            self.device = "cpu"
            self.ema = _FakeEMA(diffusion)
            self.trained = False

        def train(self):
            self.trained = True

    fake_backend = {
        "FixedSizeBinaryTableDataset": _FakeDataset,
        "BinaryDiffusion1D": _FakeDiffusion,
        "SimpleTableGenerator": _FakeModel,
        "FixedSizeTableBinaryDiffusionTrainer": _FakeTrainer,
        "get_base_model": lambda model: model,
        "get_random_labels": lambda **kwargs: __import__("torch").zeros(kwargs["n_labels"], dtype=__import__("torch").long),
    }

    monkeypatch.setattr("prelim.generators.binarydiffusion.Gen_binarydiffusion._import_backend", lambda self: fake_backend)

    x = _clustered_sample()
    generator = Gen_binarydiffusion(seed=2020).fit(x)
    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "binarydiffusion"
    assert generator.trainer_.trained is True
    assert build_generator("binarydiffusion", seed=2020).my_name() == "binarydiffusion"


def test_forestdiffusion_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeForestDiffusionModel:
        def __init__(self, X, seed, **kwargs):
            self.fit_shape_ = X.shape
            self.seed_ = seed
            self.kwargs = kwargs

        def generate(self, batch_size=None, n_t=None, X_covs=None):
            return np.arange(batch_size * 2).reshape(batch_size, 2)

    monkeypatch.setattr("prelim.generators.forestdiffusion.ForestDiffusionModel", _FakeForestDiffusionModel)

    x = _clustered_sample()
    generator = Gen_forestdiffusion(model_kwargs={"n_t": 5}, seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "forestdiffusion"
    assert generator.model_.fit_shape_ == x.shape
    assert generator.model_.seed_ == 2020
    assert generator.model_.kwargs == {"n_t": 5}
    assert build_generator("forestdiffusion", seed=2020).my_name() == "forestdiffusion"


def test_tabsyn_generator_runs_official_cli_and_returns_requested_shape(monkeypatch, tmp_path):
    repo_root = tmp_path / "tabsyn"
    (repo_root / "data" / "Info").mkdir(parents=True)
    (repo_root / "synthetic").mkdir()
    (repo_root / "main.py").write_text("", encoding="utf-8")
    (repo_root / "process_dataset.py").write_text("", encoding="utf-8")

    commands = []

    def _fake_run_command(self, command, env=None):
        del env
        commands.append(command)
        if len(command) > 2 and command[1] == "-c":
            payload = __import__("json").loads(command[3])
            if payload["module"] == "tabsyn.sample":
                sample_path = Path(payload["args"]["save_path"])
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                __import__("pandas").DataFrame(
                    {
                        "x0": np.arange(3),
                        "x1": np.arange(3, 6),
                        "__prelim_tabsyn_helper_cat": ["a", "b", "a"],
                        "__prelim_tabsyn_helper_target": np.linspace(0.0, 1.0, 3),
                    }
                ).to_csv(sample_path, index=False)

    monkeypatch.setattr("prelim.generators.tabsyn.Gen_tabsyn._run_command", _fake_run_command)

    x = _clustered_sample()
    generator = Gen_tabsyn(repo_path=repo_root, seed=2020).fit(x)

    info_path = repo_root / "data" / "Info" / f"{generator.dataset_name_}.json"
    info = __import__("json").loads(info_path.read_text(encoding="utf-8"))

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "tabsyn"
    assert build_generator("tabsyn", seed=2020).my_name() == "tabsyn"
    assert info["target_col_idx"]
    assert info["cat_col_idx"]
    assert [command[1] for command in commands] == [
        "process_dataset.py",
        "-c",
        "-c",
        "-c",
    ]
    module_payloads = [__import__("json").loads(command[3])["module"] for command in commands[1:]]
    assert module_payloads == [
        "tabsyn.vae.main",
        "tabsyn.main",
        "tabsyn.sample",
    ]


def test_tabsyn_generator_adds_helper_category_for_numeric_only_data(tmp_path):
    repo_root = tmp_path / "tabsyn"
    (repo_root / "data" / "Info").mkdir(parents=True)
    (repo_root / "main.py").write_text("", encoding="utf-8")
    (repo_root / "process_dataset.py").write_text("", encoding="utf-8")

    generator = Gen_tabsyn(repo_path=repo_root, seed=2020)
    generator.repo_root_ = repo_root
    generator.dataset_name_ = "prelim_tabsyn_test"
    generator.X_ = _clustered_sample()
    generator.columns_ = [f"x{i}" for i in range(generator.X_.shape[1])]
    generator._write_dataset_artifacts()

    info_path = repo_root / "data" / "Info" / "prelim_tabsyn_test.json"
    data_path = repo_root / "data" / "prelim_tabsyn_test" / "prelim_tabsyn_test.csv"
    info = __import__("json").loads(info_path.read_text(encoding="utf-8"))
    data = __import__("pandas").read_csv(data_path)

    assert info["target_col_idx"] == [3]
    assert info["cat_col_idx"] == [2]
    assert list(data.columns) == ["x0", "x1", "__prelim_tabsyn_helper_cat", "__prelim_tabsyn_helper_target"]


def test_bayesnet_generator_uses_backend_and_rebuilds_numeric_values(monkeypatch):
    class _FakeDag:
        def edges(self):
            return [("x0", "x1")]

    class _FakeSearch:
        def __init__(self, data):
            self.data = data

        def estimate(self, scoring_method=None, show_progress=False, **kwargs):
            self.scoring_method = scoring_method
            self.show_progress = show_progress
            self.kwargs = kwargs
            return _FakeDag()

    class _FakeBIC:
        def __init__(self, data):
            self.data = data

    class _FakeNetwork:
        def __init__(self, edges):
            self.edges_ = list(edges)
            self.nodes_ = []
            self.fit_shape_ = None
            self.estimator_ = None

        def add_nodes_from(self, nodes):
            self.nodes_.extend(nodes)

        def fit(self, data, estimator=None):
            self.fit_shape_ = data.shape
            self.estimator_ = estimator
            return self

    class _FakeSampler:
        def __init__(self, model):
            self.model = model

        def forward_sample(self, size=1, include_latents=False, seed=None, show_progress=True, partial_samples=None, n_jobs=-1):
            return __import__("pandas").DataFrame(
                {
                    "x0": [0, 1, 0, 1, 0][:size],
                    "x1": [1, 0, 1, 0, 1][:size],
                }
            )

    monkeypatch.setattr("prelim.generators.bayesnet.HillClimbSearch", _FakeSearch)
    monkeypatch.setattr("prelim.generators.bayesnet.BIC", _FakeBIC)
    monkeypatch.setattr("prelim.generators.bayesnet.DiscreteBayesianNetwork", _FakeNetwork)
    monkeypatch.setattr("prelim.generators.bayesnet.BayesianModelSampling", _FakeSampler)

    x = np.array(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [0.0, 20.0],
            [1.0, 10.0],
        ]
    )
    generator = Gen_bayesnet(max_bins=4, seed=2020).fit(x)

    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "bayesnet"
    assert generator.model_.fit_shape_ == x.shape
    assert generator.model_.nodes_ == ["x0", "x1"]
    assert set(np.unique(sample[:, 0])).issubset({0.0, 1.0})
    assert set(np.unique(sample[:, 1])).issubset({10.0, 20.0})
    assert build_generator("bayesnet", seed=2020).my_name() == "bayesnet"


def test_perfect_returns_subset_without_replacement_when_possible():
    x = _clustered_sample()
    generator = Gen_perfect().fit(x)

    sample = generator.sample(n_samples=10)

    assert sample.shape == (10, x.shape[1])
    assert len(np.unique(sample, axis=0)) == 10
    assert set(map(tuple, sample)).issubset(set(map(tuple, x)))
    assert generator.my_name() == "perfect"


def test_perfect_warns_and_returns_complete_set_when_too_many_points_requested():
    x = _clustered_sample()
    generator = Gen_perfect().fit(x)

    with pytest.warns(UserWarning, match="Too many points are requested"):
        sample = generator.sample(n_samples=len(x) + 1)

    assert np.array_equal(sample, x)


def test_noise_perturbs_points_within_expected_per_feature_range():
    x = _grid_sample()
    generator = Gen_noise(scale=0.3).fit(x)

    sample = generator.sample(n_samples=1000)

    assert sample.shape == x.shape
    assert generator.my_name() == "noise"
    max_delta = 0.3
    assert np.all(np.abs(sample - x) <= max_delta + 1e-12)


def test_randu_samples_stay_within_feature_bounds():
    x = _clustered_sample()
    generator = Gen_randu().fit(x)

    sample = generator.sample(n_samples=25)

    assert sample.shape == (25, x.shape[1])
    assert np.all(sample >= x.min(axis=0))
    assert np.all(sample <= x.max(axis=0))
    assert generator.my_name() == "randu"


def test_randn_learns_location_and_covariance_and_samples_requested_shape():
    x = _clustered_sample()
    generator = Gen_randn(seed=2020).fit(x)

    sample = generator.sample(n_samples=25)

    assert sample.shape == (25, x.shape[1])
    assert generator.location_.shape == (x.shape[1],)
    assert generator.covariance_.shape == (x.shape[1], x.shape[1])
    assert generator.my_name() == "randn"


@pytest.mark.parametrize(
    ("generator_a", "generator_b", "sample_kwargs"),
    [
        (Gen_randn(seed=2020), Gen_randn(seed=2020), {"n_samples": 25}),
        (Gen_randu(seed=2020), Gen_randu(seed=2020), {"n_samples": 25}),
        (Gen_noise(scale=0.3, seed=2020), Gen_noise(scale=0.3, seed=2020), {"n_samples": 25}),
        (Gen_perfect(seed=2020), Gen_perfect(seed=2020), {"n_samples": 10}),
        (Gen_kdeb(knn=5, seed=2020), Gen_kdeb(knn=5, seed=2020), {"n_samples": 20}),
        (Gen_kdebw(seed=2020), Gen_kdebw(seed=2020), {"n_samples": 20}),
        (Gen_kdebwhl(seed=2020), Gen_kdebwhl(seed=2020), {"n_samples": 20}),
        (Gen_kdebwm(seed=2020), Gen_kdebwm(seed=2020), {"n_samples": 20}),
        (Gen_munge(local_var=1, p_swap=0.5, seed=2020), Gen_munge(local_var=1, p_swap=0.5, seed=2020), {"n_samples": 20}),
        (Gen_rfdens(seed=2020), Gen_rfdens(seed=2020), {"n_samples": 20}),
        (Gen_vva_proba(seed=2020), Gen_vva_proba(seed=2020), {"r": 1.0}),
    ],
)
def test_seeded_generators_are_reproducible(generator_a, generator_b, sample_kwargs):
    x = _clustered_sample()
    y = np.concatenate((np.zeros(40, dtype=int), np.ones(40, dtype=int)))

    if isinstance(generator_a, Gen_rfdens):
        generator_a.fit(x, y)
        generator_b.fit(x, y)
    elif isinstance(generator_a, Gen_vva_proba):
        generator_a.fit(x, _LinearProbabilityMeta())
        generator_b.fit(x, _LinearProbabilityMeta())
    else:
        generator_a.fit(x)
        generator_b.fit(x)

    sample_a = generator_a.sample(**sample_kwargs)
    sample_b = generator_b.sample(**sample_kwargs)

    assert np.allclose(sample_a, sample_b)


def test_kde_hard_limits_samples_within_observed_min_max():
    x = _clustered_sample()
    generator = Gen_kdebwhl().fit(x)

    sample = generator.sample(n_samples=50)

    assert sample.shape == (50, x.shape[1])
    assert np.all(sample >= x.min(axis=0))
    assert np.all(sample <= x.max(axis=0))
    assert generator.my_name() == "kdebwhl"


def test_kde_bandwidth_generator_returns_requested_shape():
    x = _clustered_sample()
    generator = Gen_kdebw().fit(x)

    sample = generator.sample(n_samples=50)

    assert sample.shape == (50, x.shape[1])
    assert generator.my_name() == "kdebw"


def test_kdeb_rejects_knn_greater_than_or_equal_to_dataset_size():
    x = _clustered_sample()

    with pytest.raises(RuntimeError, match="dataset is too small"):
        Gen_kdeb(knn=len(x)).fit(x)


def test_kdeb_samples_requested_shape_for_knn_zero_example():
    x = np.array([[0.0, 0.0]])
    generator = Gen_kdeb(knn=0).fit(x)

    sample = generator.sample(n_samples=20)

    assert sample.shape == (20, x.shape[1])
    assert generator.dist_ == 1
    assert generator.my_name() == "kdeb"


def test_kdebwm_invalid_method_is_rejected():
    with pytest.raises(ValueError, match="either scott or silverman"):
        Gen_kdebwm(method="invalid")


def test_kdebwm_samples_requested_shape():
    x = _clustered_sample()
    generator = Gen_kdebwm().fit(x)

    sample = generator.sample(n_samples=30)

    assert sample.shape == (30, x.shape[1])
    assert len(generator.model_) == x.shape[1]
    assert generator.my_name() == "kdebwm"


def test_munge_rejects_too_small_p_swap():
    with pytest.raises(ValueError, match="p_swap parameter is too small"):
        Gen_munge(p_swap=0.001)


def test_munge_generates_unique_requested_number_of_rows():
    x = _clustered_sample()
    generator = Gen_munge(local_var=1, p_swap=0.5, seed=2020).fit(x)

    sample = generator.sample(n_samples=60)

    assert sample.shape == (60, x.shape[1])
    assert len(np.unique(sample, axis=0)) == 60
    assert generator.my_name() == "munge"


@pytest.mark.parametrize(
    ("generator_cls", "expected_name"),
    [
        (Gen_smote, "smote"),
        (Gen_adasyn, "adasyn"),
    ],
)
def test_smote_like_generators_warn_when_requested_size_is_smaller_than_train_set(generator_cls, expected_name):
    x = _clustered_sample()
    generator = generator_cls().fit(x)

    with pytest.warns(UserWarning):
        sample = generator.sample(n_samples=20)

    assert sample.shape == (20, x.shape[1])
    assert generator.my_name() == expected_name


@pytest.mark.parametrize(
    ("generator_cls", "expected_name"),
    [
        (Gen_smote, "smote"),
        (Gen_adasyn, "adasyn"),
    ],
)
def test_smote_like_generators_return_requested_shape_on_example_style_input(generator_cls, expected_name):
    x = _clustered_sample()
    generator = generator_cls().fit(x)

    sample = generator.sample(n_samples=120)

    assert sample.shape == (120, x.shape[1])
    assert generator.my_name() == expected_name


def test_rerx_returns_only_correctly_predicted_rows():
    x, y = _labeled_clustered_sample()
    generator = Gen_rerx().fit(x, y, _AllOnesMeta())

    sample = generator.sample(n_samples=5)

    assert np.array_equal(sample, x[y == 1])
    assert generator.my_name() == "rerx"


def test_rfdens_fit_populates_boxes_and_sample_stays_within_global_bounds():
    x, y = _labeled_clustered_sample()
    generator = Gen_rfdens()

    generator.fit(x, y)
    sample = generator.sample(n_samples=25)

    assert len(generator.boxes_) > 0
    assert len(generator.boxes_) == len(generator.nsamples_)
    assert sample.shape == (25, x.shape[1])
    assert np.all(sample >= x.min(axis=0))
    assert np.all(sample <= x.max(axis=0))
    assert generator.my_name() == "cmmrf"


@pytest.mark.parametrize(
    ("generator_cls", "expected_name"),
    [
        (Gen_gmm, "gmmcv"),
        (Gen_gmmbic, "gmm"),
        (Gen_gmmbical, "gmmal"),
    ],
)
def test_gmm_family_generates_requested_shape(generator_cls, expected_name):
    x = _clustered_sample()
    params = {"covariance_type": ["diag"], "n_components": [1, 2]} if generator_cls is not Gen_gmmbical else {"n_components": [1, 2]}
    generator = generator_cls(params=params).fit(x)

    sample = generator.sample(n_samples=20)

    assert sample.shape == (20, x.shape[1])
    assert generator.my_name() == expected_name


def test_vva_proba_returns_empty_sample_for_r_zero_and_out_of_range_r_fails():
    x = _clustered_sample()
    generator = Gen_vva_proba().fit(x, _LinearProbabilityMeta())

    empty = generator.sample(r=0)

    assert empty.shape == (0, x.shape[1])
    with pytest.raises(ValueError, match="from 0 to 2.5"):
        generator.sample(r=3.0)
    assert generator.my_name() == "vva"


def test_vva_proba_disables_generation_when_all_predictions_are_on_one_side():
    x = _clustered_sample()
    generator = Gen_vva_proba().fit(x, _SingleSideProbabilityMeta())

    assert generator.will_generate() is False
    assert generator.sample(r=1.0).shape == (0, x.shape[1])


def test_vva_proba_generates_requested_number_of_boundary_points():
    x = _clustered_sample()
    generator = Gen_vva_proba(rho=0.2).fit(x, _LinearProbabilityMeta())

    sample = generator.sample(r=1.0)

    assert generator.will_generate() is True
    assert sample.shape == (len(x), x.shape[1])


def test_vva_legacy_disables_generation_when_all_predictions_are_on_one_side():
    x = _clustered_sample()
    generator = Gen_vva_legacy().fit(x, _SingleSideScoreMeta())

    assert generator.will_generate() is False
    assert generator.sample(r=1.0).shape == (0, x.shape[1])


def test_vva_legacy_generates_requested_number_of_boundary_points():
    x = _clustered_sample()
    generator = Gen_vva_legacy(rho=0.2).fit(x, _LinearScoreMeta())

    sample = generator.sample(r=1.0)

    assert generator.will_generate() is True
    assert sample.shape == (len(x), x.shape[1])
