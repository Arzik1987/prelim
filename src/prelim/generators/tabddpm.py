import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseGenerator

_RUNNER = textwrap.dedent(
    """
    import json
    import os
    import sys

    payload = json.loads(sys.argv[1])
    if payload["action"] == "train":
        from train import train
        train(**payload["kwargs"])
    elif payload["action"] == "sample":
        from sample import sample
        sample(**payload["kwargs"])
    else:
        raise ValueError(payload["action"])
    """
)


class Gen_tabddpm(BaseGenerator):
    def __init__(
        self,
        repo_path: str | os.PathLike | None = None,
        python_executable: str | None = None,
        train_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        keep_artifacts: bool = False,
        seed=2020,
    ):
        super().__init__("tabddpm", seed=seed)
        self.repo_path_ = None if repo_path is None else Path(repo_path).expanduser().resolve()
        self.python_executable_ = python_executable or sys.executable
        self.train_kwargs_ = {} if train_kwargs is None else dict(train_kwargs)
        self.sample_kwargs_ = {} if sample_kwargs is None else dict(sample_kwargs)
        self.keep_artifacts_ = keep_artifacts
        self.X_ = None
        self.repo_root_ = None
        self.run_root_ = None
        self.data_root_ = None
        self.output_root_ = None
        self.column_specs_ = None

    def fit(self, X, y=None, metamodel=None):
        del y, metamodel
        self._cleanup_artifacts()
        self.X_ = np.asarray(X).copy()
        self.repo_root_ = self._resolve_repo_root()
        self.run_root_ = Path(tempfile.mkdtemp(prefix="prelim-tabddpm-"))
        self.data_root_ = self.run_root_ / "data"
        self.output_root_ = self.run_root_ / "output"
        self.data_root_.mkdir(parents=True, exist_ok=True)
        self.output_root_.mkdir(parents=True, exist_ok=True)

        self.column_specs_ = self._analyze_columns(self.X_)
        self._write_dataset_artifacts()
        self._run_action("train", self._build_train_kwargs())
        return self

    def sample(self, n_samples=1):
        if self.output_root_ is None or self.data_root_ is None or self.column_specs_ is None:
            raise RuntimeError("Gen_tabddpm.sample called before fit")

        self._run_action("sample", self._build_sample_kwargs(n_samples))
        sampled = self._load_sampled_rows()
        if len(sampled) == 0:
            raise RuntimeError("TabDDPM returned zero sampled rows")
        if len(sampled) >= n_samples:
            return sampled[:n_samples].copy()

        indices = self.rng_.choice(len(sampled), size=n_samples, replace=True)
        return sampled[indices].copy()

    def _resolve_repo_root(self):
        repo_root = self.repo_path_
        if repo_root is None:
            env_path = os.environ.get("TABDDPM_REPO_PATH")
            if env_path:
                repo_root = Path(env_path).expanduser().resolve()
        if repo_root is None:
            raise RuntimeError(
                "TabDDPM requires a local checkout of the official repository. "
                "Set TABDDPM_REPO_PATH or pass repo_path=... to Gen_tabddpm."
            )
        if not repo_root.exists():
            raise FileNotFoundError(f"TabDDPM repo path does not exist: {repo_root}")
        if not (repo_root / "scripts" / "train.py").exists() or not (repo_root / "scripts" / "sample.py").exists():
            raise FileNotFoundError(f"TabDDPM repo path does not look like the official repository: {repo_root}")
        return repo_root

    def _analyze_columns(self, X):
        frame = pd.DataFrame(X)
        specs = []
        for index in range(frame.shape[1]):
            series = frame.iloc[:, index]
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().all():
                values = numeric.astype(float).to_numpy()
                specs.append({"kind": "num", "index": index, "values": values})
            else:
                values = series.astype(str).to_numpy()
                specs.append({"kind": "cat", "index": index, "values": values})
        return specs

    def _write_dataset_artifacts(self):
        numeric_columns = [spec["values"] for spec in self.column_specs_ if spec["kind"] == "num"]
        categorical_columns = [spec["values"] for spec in self.column_specs_ if spec["kind"] == "cat"]

        X_num = np.column_stack(numeric_columns) if numeric_columns else None
        X_cat = np.column_stack(categorical_columns) if categorical_columns else None
        y = self._build_helper_target(X_num, X_cat)

        train_end = max(2, int(len(y) * 0.8))
        val_end = max(train_end + 1, int(len(y) * 0.9))
        val_end = min(val_end, len(y) - 1)
        splits = {
            "train": slice(0, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, len(y)),
        }

        for split, split_slice in splits.items():
            np.save(self.data_root_ / f"y_{split}.npy", y[split_slice])
            if X_num is not None:
                np.save(self.data_root_ / f"X_num_{split}.npy", X_num[split_slice])
            if X_cat is not None:
                np.save(self.data_root_ / f"X_cat_{split}.npy", X_cat[split_slice].astype(str))

        info = {
            "task_type": "binclass",
            "n_classes": 2,
            "train_size": train_end,
            "val_size": val_end - train_end,
            "test_size": len(y) - val_end,
            "n_num_features": 0 if X_num is None else X_num.shape[1],
            "n_cat_features": 0 if X_cat is None else X_cat.shape[1],
        }
        with open(self.data_root_ / "info.json", "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)

    def _build_helper_target(self, X_num, X_cat):
        if X_num is not None and X_num.shape[1] > 0:
            source = X_num[:, 0]
            threshold = np.median(source)
            y = (source >= threshold).astype(np.int64)
        elif X_cat is not None and X_cat.shape[1] > 0:
            codes, _ = pd.factorize(X_cat[:, 0])
            y = (codes % 2).astype(np.int64)
        else:
            y = (np.arange(len(self.X_)) % 2).astype(np.int64)

        if len(np.unique(y)) < 2 and len(y) > 1:
            y = (np.arange(len(y)) % 2).astype(np.int64)
        return y

    def _build_train_kwargs(self):
        num_numerical_features = sum(spec["kind"] == "num" for spec in self.column_specs_)
        model_params = {
            "num_classes": 2,
            "is_y_cond": True,
            "rtdl_params": self.train_kwargs_.get("rtdl_params", {"d_layers": [128, 128], "dropout": 0.0}),
        }
        kwargs = {
            "parent_dir": str(self.output_root_),
            "real_data_path": str(self.data_root_),
            "steps": self.train_kwargs_.get("steps", int(os.environ.get("PRELIM_TABDDPM_STEPS", "100"))),
            "lr": self.train_kwargs_.get("lr", 0.001),
            "weight_decay": self.train_kwargs_.get("weight_decay", 0.0),
            "batch_size": self.train_kwargs_.get("batch_size", 256),
            "model_type": self.train_kwargs_.get("model_type", "mlp"),
            "model_params": model_params,
            "num_timesteps": self.train_kwargs_.get("num_timesteps", int(os.environ.get("PRELIM_TABDDPM_TIMESTEPS", "100"))),
            "gaussian_loss_type": self.train_kwargs_.get("gaussian_loss_type", "mse"),
            "scheduler": self.train_kwargs_.get("scheduler", "cosine"),
            "T_dict": self.train_kwargs_.get(
                "T_dict",
                {
                    "seed": self.seed_,
                    "normalization": "standard",
                    "num_nan_policy": None,
                    "cat_nan_policy": None,
                    "cat_min_frequency": None,
                    "cat_encoding": None,
                    "y_policy": "default",
                },
            ),
            "num_numerical_features": num_numerical_features,
            "device": "cpu",
            "seed": self.seed_,
            "change_val": False,
        }
        return kwargs

    def _build_sample_kwargs(self, n_samples):
        train_kwargs = self._build_train_kwargs()
        kwargs = {
            "parent_dir": str(self.output_root_),
            "real_data_path": str(self.data_root_),
            "batch_size": self.sample_kwargs_.get("batch_size", max(64, n_samples)),
            "num_samples": n_samples,
            "model_type": train_kwargs["model_type"],
            "model_params": train_kwargs["model_params"],
            "model_path": str(self.output_root_ / "model.pt"),
            "num_timesteps": train_kwargs["num_timesteps"],
            "gaussian_loss_type": train_kwargs["gaussian_loss_type"],
            "scheduler": train_kwargs["scheduler"],
            "T_dict": train_kwargs["T_dict"],
            "num_numerical_features": train_kwargs["num_numerical_features"],
            "disbalance": None,
            "device": "cpu",
            "seed": self.seed_,
            "change_val": False,
        }
        return kwargs

    def _run_action(self, action, kwargs):
        payload = {"action": action, "kwargs": kwargs}
        env = os.environ.copy()
        pythonpath_entries = [str(self.repo_root_), str(self.repo_root_ / "scripts")]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "prelim-tabddpm-mpl"))
        try:
            subprocess.run(
                [self.python_executable_, "-c", _RUNNER, json.dumps(payload)],
                cwd=self.repo_root_,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            output = "\n".join(part for part in [exc.stdout.strip(), exc.stderr.strip()] if part)
            raise RuntimeError(f"TabDDPM {action} failed\n{output}") from exc

    def _load_sampled_rows(self):
        X_num = None
        X_cat = None
        if (self.output_root_ / "X_num_train.npy").exists():
            X_num = np.load(self.output_root_ / "X_num_train.npy", allow_pickle=True)
        if (self.output_root_ / "X_cat_train.npy").exists():
            X_cat = np.load(self.output_root_ / "X_cat_train.npy", allow_pickle=True)

        num_cursor = 0
        cat_cursor = 0
        columns = []
        for spec in self.column_specs_:
            if spec["kind"] == "num":
                columns.append(np.asarray(X_num[:, num_cursor], dtype=float))
                num_cursor += 1
            else:
                columns.append(np.asarray(X_cat[:, cat_cursor]).astype(str))
                cat_cursor += 1
        return np.column_stack(columns)

    def _cleanup_artifacts(self):
        if self.keep_artifacts_ or self.run_root_ is None:
            return
        shutil.rmtree(self.run_root_, ignore_errors=True)

    def __del__(self):
        self._cleanup_artifacts()
