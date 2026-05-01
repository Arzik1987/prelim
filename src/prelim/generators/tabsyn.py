import json
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseGenerator

_HELPER_CAT_COLUMN = "__prelim_tabsyn_helper_cat"
_HELPER_TARGET_COLUMN = "__prelim_tabsyn_helper_target"

_MODULE_RUNNER = textwrap.dedent(
    """
    import importlib
    import json
    import sys
    from argparse import Namespace

    payload = json.loads(sys.argv[1])
    module = importlib.import_module(payload["module"])
    args = Namespace(**payload["args"])

    if payload["module"] in {"tabsyn.vae.main", "tabsyn.main"}:
        scheduler_class = module.ReduceLROnPlateau

        def _compat_scheduler(*scheduler_args, **scheduler_kwargs):
            scheduler_kwargs.pop("verbose", None)
            return scheduler_class(*scheduler_args, **scheduler_kwargs)

        module.ReduceLROnPlateau = _compat_scheduler

    if payload["module"] == "tabsyn.sample":
        diffusion_sample = module.sample
        device = payload["args"]["device"]
        steps = payload["args"].get("steps")

        def _sample_with_device(net, num_samples, dim, *unused_args, **unused_kwargs):
            if steps is None:
                return diffusion_sample(net, num_samples, dim, device=device)
            return diffusion_sample(net, num_samples, dim, num_steps=steps, device=device)

        module.sample = _sample_with_device

    module.main(args)
    """
)


class Gen_tabsyn(BaseGenerator):
    def __init__(
        self,
        repo_path: str | os.PathLike | None = None,
        python_executable: str | None = None,
        train_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        keep_artifacts: bool = False,
        seed=2020,
    ):
        super().__init__("tabsyn", seed=seed)
        self.repo_path_ = None if repo_path is None else Path(repo_path).expanduser().resolve()
        self.python_executable_ = python_executable or sys.executable
        self.train_kwargs_ = {} if train_kwargs is None else dict(train_kwargs)
        self.sample_kwargs_ = {} if sample_kwargs is None else dict(sample_kwargs)
        self.keep_artifacts_ = keep_artifacts
        self.X_ = None
        self.columns_ = None
        self.dataset_name_ = None
        self.sample_path_ = None
        self.repo_root_ = None

    def fit(self, X, y=None, metamodel=None):
        del y, metamodel
        self._cleanup_repo_artifacts()
        self.X_ = np.asarray(X).copy()
        self.columns_ = [f"x{i}" for i in range(self.X_.shape[1])]
        self.repo_root_ = self._resolve_repo_root()
        self.dataset_name_ = f"prelim_tabsyn_{uuid.uuid4().hex}"
        self.sample_path_ = self.repo_root_ / "synthetic" / self.dataset_name_ / "prelim_sample.csv"

        self._write_dataset_artifacts()
        self._run_process_dataset()
        self._run_module("tabsyn.vae.main", self._build_vae_args())
        self._run_module("tabsyn.main", self._build_diffusion_args())
        return self

    def sample(self, n_samples=1):
        if self.repo_root_ is None or self.dataset_name_ is None or self.sample_path_ is None:
            raise RuntimeError("Gen_tabsyn.sample called before fit")

        self._run_module("tabsyn.sample", self._build_sample_args())
        sampled = pd.read_csv(self.sample_path_)
        sampled = sampled.loc[:, self.columns_]

        if len(sampled) == 0:
            raise RuntimeError("TabSyn returned zero sampled rows")
        if n_samples <= len(sampled):
            return sampled.iloc[:n_samples].to_numpy(copy=True)

        indices = self.rng_.choice(len(sampled), size=n_samples, replace=True)
        return sampled.iloc[indices].to_numpy(copy=True)

    def _resolve_repo_root(self):
        repo_root = self.repo_path_
        if repo_root is None:
            env_path = os.environ.get("TABSYN_REPO_PATH")
            if env_path:
                repo_root = Path(env_path).expanduser().resolve()

        if repo_root is None:
            raise RuntimeError(
                "TabSyn requires a local checkout of the official repository. "
                "Set TABSYN_REPO_PATH or pass repo_path=... to Gen_tabsyn."
            )
        if not repo_root.exists():
            raise FileNotFoundError(f"TabSyn repo path does not exist: {repo_root}")
        if not (repo_root / "main.py").exists() or not (repo_root / "process_dataset.py").exists():
            raise FileNotFoundError(f"TabSyn repo path does not look like the official repository: {repo_root}")
        return repo_root

    def _write_dataset_artifacts(self):
        dataset_dir = self.repo_root_ / "data" / self.dataset_name_
        info_dir = self.repo_root_ / "data" / "Info"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        info_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(self.X_, columns=self.columns_)
        num_col_idx = []
        cat_col_idx = []
        for index, column in enumerate(self.columns_):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().all():
                frame[column] = numeric.astype(float)
                num_col_idx.append(index)
            else:
                frame[column] = frame[column].astype(str)
                cat_col_idx.append(index)

        if not cat_col_idx:
            frame[_HELPER_CAT_COLUMN] = np.where(np.arange(len(frame)) % 2 == 0, "a", "b")
            cat_col_idx.append(len(frame.columns) - 1)

        helper_target = np.linspace(0.0, 1.0, len(frame), dtype=float)
        if num_col_idx:
            helper_target = helper_target + pd.to_numeric(frame.iloc[:, num_col_idx[0]], errors="coerce").to_numpy() * 0.01
        frame[_HELPER_TARGET_COLUMN] = helper_target
        target_col_idx = [len(frame.columns) - 1]

        data_path = dataset_dir / f"{self.dataset_name_}.csv"
        info_path = info_dir / f"{self.dataset_name_}.json"
        frame.to_csv(data_path, index=False)

        info = {
            "name": self.dataset_name_,
            "task_type": "regression",
            "header": "infer",
            "column_names": list(frame.columns),
            "num_col_idx": num_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": target_col_idx,
            "file_type": "csv",
            "data_path": str(data_path.relative_to(self.repo_root_)),
            "test_path": None,
        }
        with open(info_path, "w", encoding="utf-8") as handle:
            json.dump(info, handle, indent=2)

    def _build_vae_args(self):
        return {
            "dataname": self.dataset_name_,
            "gpu": -1,
            "device": "cpu",
            "max_beta": self.train_kwargs_.get("max_beta", 1e-2),
            "min_beta": self.train_kwargs_.get("min_beta", 1e-5),
            "lambd": self.train_kwargs_.get("lambd", 0.7),
        }

    def _build_diffusion_args(self):
        return {
            "dataname": self.dataset_name_,
            "gpu": -1,
            "device": "cpu",
        }

    def _build_sample_args(self):
        return {
            "dataname": self.dataset_name_,
            "gpu": -1,
            "device": "cpu",
            "epoch": self.sample_kwargs_.get("epoch"),
            "steps": self.sample_kwargs_.get("steps"),
            "save_path": str(self.sample_path_),
        }

    def _run_process_dataset(self):
        self._run_command(
            [
                self.python_executable_,
                "process_dataset.py",
                "--dataname",
                self.dataset_name_,
            ]
        )

    def _run_module(self, module_name, module_args):
        payload = {"module": module_name, "args": module_args}
        self._run_command(
            [
                self.python_executable_,
                "-c",
                _MODULE_RUNNER,
                json.dumps(payload),
            ],
            env=self._build_env(),
        )

    def _build_env(self):
        env = os.environ.copy()
        pythonpath_entries = [str(self.repo_root_)]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "prelim-tabsyn-mpl"))
        return env

    def _run_command(self, command, env=None):
        try:
            subprocess.run(
                command,
                cwd=self.repo_root_,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            output = "\n".join(part for part in [exc.stdout.strip(), exc.stderr.strip()] if part)
            message = output or str(exc)
            raise RuntimeError(f"TabSyn command failed: {' '.join(command)}\n{message}") from exc

    def _cleanup_repo_artifacts(self):
        if self.keep_artifacts_ or not self.dataset_name_ or self.repo_root_ is None:
            return
        if not self.dataset_name_.startswith("prelim_tabsyn_"):
            return

        paths = [
            self.repo_root_ / "data" / self.dataset_name_,
            self.repo_root_ / "data" / "Info" / f"{self.dataset_name_}.json",
            self.repo_root_ / "synthetic" / self.dataset_name_,
            self.repo_root_ / "tabsyn" / "ckpt" / self.dataset_name_,
            self.repo_root_ / "tabsyn" / "vae" / "ckpt" / self.dataset_name_,
        ]
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)

    def __del__(self):
        self._cleanup_repo_artifacts()
