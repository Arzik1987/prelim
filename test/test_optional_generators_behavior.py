from pathlib import Path

import numpy as np

from prelim.generators import Gen_binarydiffusion
from prelim.generators import Gen_great
from prelim.generators import Gen_tabddpm
from prelim.generators import Gen_tabsyn
from prelim.generators import build_generator


def _clustered_sample():
    rng = np.random.RandomState(2020)
    x1 = rng.multivariate_normal([0.0, 0.0], np.eye(2) * 0.2, 40)
    x2 = rng.multivariate_normal([3.0, 3.0], np.eye(2) * 0.2, 40)
    return np.vstack((x1, x2))


def test_great_generator_uses_backend_and_returns_requested_shape(monkeypatch):
    class _FakeGReaT:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.fit_shape_ = None

        def fit(self, data):
            self.fit_shape_ = data.shape

        def sample(self, n_samples, **kwargs):
            del kwargs
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
            self.targets_binary = __import__("torch").zeros(len(table), dtype=__import__("torch").long)

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

    def _fake_import_backend(self):
        self._backend = fake_backend
        return fake_backend

    monkeypatch.setattr("prelim.generators.binarydiffusion.Gen_binarydiffusion._import_backend", _fake_import_backend)

    x = _clustered_sample()
    generator = Gen_binarydiffusion(seed=2020).fit(x)
    sample = generator.sample(n_samples=5)

    assert sample.shape == (5, x.shape[1])
    assert generator.my_name() == "binarydiffusion"
    assert generator.trainer_.trained is True
    assert build_generator("binarydiffusion", seed=2020).my_name() == "binarydiffusion"


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
