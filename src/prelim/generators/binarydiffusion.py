import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .base import BaseGenerator


class _NoOpLogger:
    def log(self, *_args, **_kwargs):
        return None


class Gen_binarydiffusion(BaseGenerator):
    def __init__(
        self,
        repo_path: str | os.PathLike | None = None,
        model_kwargs: dict | None = None,
        diffusion_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        sample_kwargs: dict | None = None,
        keep_artifacts: bool = False,
        seed=2020,
    ):
        super().__init__("binarydiffusion", seed=seed)
        self.repo_path_ = None if repo_path is None else Path(repo_path).expanduser().resolve()
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.diffusion_kwargs_ = {} if diffusion_kwargs is None else dict(diffusion_kwargs)
        self.trainer_kwargs_ = {} if trainer_kwargs is None else dict(trainer_kwargs)
        self.sample_kwargs_ = {} if sample_kwargs is None else dict(sample_kwargs)
        self.keep_artifacts_ = keep_artifacts
        self.X_ = None
        self.dataset_ = None
        self.trainer_ = None
        self.run_root_ = None
        self.columns_ = None
        self.numeric_columns_ = None
        self.categorical_columns_ = None
        self._backend = None

    def fit(self, X, y=None, metamodel=None):
        del y, metamodel
        self._cleanup_artifacts()
        self.X_ = np.asarray(X, dtype=object).copy()
        self.columns_ = [f"x{i}" for i in range(self.X_.shape[1])]
        frame = pd.DataFrame(self.X_, columns=self.columns_)
        self.numeric_columns_, self.categorical_columns_ = self._split_columns(frame)
        backend = self._import_backend()

        task = "classification"
        target_column = "__prelim_target"
        helper_target = self._build_helper_target(frame)
        table = frame.copy()
        table[target_column] = helper_target

        dataset = backend["FixedSizeBinaryTableDataset"](
            table=table,
            target_column=target_column,
            split_feature_target=True,
            task=task,
            numerical_columns=self.numeric_columns_,
            categorical_columns=self.categorical_columns_,
        )
        dataset.targets_binary = dataset.targets_binary.long()

        classifier_free_guidance = self.trainer_kwargs_.get("classifier_free_guidance", False)
        target_mode = self.diffusion_kwargs_.get("target", "two_way")
        model = backend["SimpleTableGenerator"](
            data_dim=dataset.row_size,
            dim=self.model_kwargs_.get("dim", 64),
            n_res_blocks=self.model_kwargs_.get("n_res_blocks", 2),
            out_dim=dataset.row_size * 2 if target_mode == "two_way" else dataset.row_size,
            task=task,
            conditional=dataset.conditional,
            n_classes=dataset.n_classes,
            classifier_free_guidance=classifier_free_guidance,
        ).to("cpu")

        diffusion = backend["BinaryDiffusion1D"](
            denoise_model=model,
            schedule=self.diffusion_kwargs_.get("schedule", "quad"),
            n_timesteps=self.diffusion_kwargs_.get("n_timesteps", int(os.environ.get("PRELIM_BINARYDIFFUSION_TIMESTEPS", "50"))),
            target=target_mode,
        ).to("cpu")

        self.run_root_ = Path(tempfile.mkdtemp(prefix="prelim-binarydiffusion-"))
        trainer = backend["FixedSizeTableBinaryDiffusionTrainer"](
            diffusion=diffusion,
            dataset=dataset,
            train_num_steps=self.trainer_kwargs_.get("train_num_steps", int(os.environ.get("PRELIM_BINARYDIFFUSION_STEPS", "50"))),
            log_every=self.trainer_kwargs_.get("log_every", 1000),
            save_every=self.trainer_kwargs_.get("save_every", 1000000),
            save_num_samples=self.trainer_kwargs_.get("save_num_samples", 8),
            max_grad_norm=self.trainer_kwargs_.get("max_grad_norm"),
            gradient_accumulate_every=self.trainer_kwargs_.get("gradient_accumulate_every", 1),
            ema_decay=self.trainer_kwargs_.get("ema_decay", 0.995),
            ema_update_every=self.trainer_kwargs_.get("ema_update_every", 10),
            lr=self.trainer_kwargs_.get("lr", 3e-4),
            opt_type=self.trainer_kwargs_.get("opt_type", "adam"),
            opt_params=self.trainer_kwargs_.get("opt_params"),
            batch_size=self.trainer_kwargs_.get("batch_size", 32),
            dataloader_workers=self.trainer_kwargs_.get("dataloader_workers", 0),
            classifier_free_guidance=classifier_free_guidance,
            zero_token_probability=self.trainer_kwargs_.get("zero_token_probability", 0.0),
            logger=_NoOpLogger(),
            results_folder=self.run_root_,
        )

        trainer.train()
        self.dataset_ = dataset
        self.trainer_ = trainer
        return self

    def sample(self, n_samples=1):
        if self.dataset_ is None or self.trainer_ is None or self._backend is None:
            raise RuntimeError("Gen_binarydiffusion.sample called before fit")

        backend = self._backend
        diffusion = backend["get_base_model"](self.trainer_.ema.ema_model)
        diffusion.eval()

        labels = backend["get_random_labels"](
            conditional=self.dataset_.conditional,
            task=self.dataset_.task,
            n_classes=self.dataset_.n_classes,
            classifier_free_guidance=self.trainer_kwargs_.get("classifier_free_guidance", False),
            n_labels=n_samples,
            device=self.trainer_.device,
        )
        rows = diffusion.sample(
            n=n_samples,
            y=labels,
            threshold=self.sample_kwargs_.get("threshold", 0.5),
            strategy=self.sample_kwargs_.get("strategy", "target"),
            timesteps=self.sample_kwargs_.get("timesteps"),
        )

        if self.dataset_.conditional:
            if self.trainer_kwargs_.get("classifier_free_guidance", False):
                labels = torch.argmax(labels, dim=1).detach()
            rows_df, _ = self.dataset_.transformation.inverse_transform(rows, labels)
        else:
            rows_df = self.dataset_.transformation.inverse_transform(rows)

        rows_df = rows_df.loc[:, self.columns_]
        return rows_df.to_numpy()

    def _split_columns(self, frame):
        numeric = []
        categorical = []
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce")
            if series.notna().all():
                frame[column] = series.astype(float)
                numeric.append(column)
            else:
                frame[column] = frame[column].astype(str)
                categorical.append(column)
        return numeric, categorical

    def _build_helper_target(self, frame):
        if self.numeric_columns_:
            source = pd.to_numeric(frame[self.numeric_columns_[0]], errors="coerce").to_numpy()
            return (source >= np.median(source)).astype(int)
        codes, _ = pd.factorize(frame[self.categorical_columns_[0]].astype(str))
        target = (codes % 2).astype(int)
        if len(np.unique(target)) < 2 and len(target) > 1:
            target = (np.arange(len(target)) % 2).astype(int)
        return target

    def _import_backend(self):
        if self._backend is not None:
            return self._backend

        repo_root = self.repo_path_
        if repo_root is None:
            env_path = os.environ.get("BINARYDIFFUSION_REPO_PATH")
            if env_path:
                repo_root = Path(env_path).expanduser().resolve()
        if repo_root is not None and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        sys.modules.setdefault("wandb", types.SimpleNamespace(init=lambda *args, **kwargs: _NoOpLogger()))

        from binary_diffusion_tabular.dataset import FixedSizeBinaryTableDataset
        from binary_diffusion_tabular.diffusion import BinaryDiffusion1D
        from binary_diffusion_tabular.model import SimpleTableGenerator
        from binary_diffusion_tabular.trainer import FixedSizeTableBinaryDiffusionTrainer
        from binary_diffusion_tabular.utils import get_base_model, get_random_labels

        self._backend = {
            "FixedSizeBinaryTableDataset": FixedSizeBinaryTableDataset,
            "BinaryDiffusion1D": BinaryDiffusion1D,
            "SimpleTableGenerator": SimpleTableGenerator,
            "FixedSizeTableBinaryDiffusionTrainer": FixedSizeTableBinaryDiffusionTrainer,
            "get_base_model": get_base_model,
            "get_random_labels": get_random_labels,
        }
        return self._backend

    def _cleanup_artifacts(self):
        if self.keep_artifacts_ or self.run_root_ is None:
            return
        import shutil

        shutil.rmtree(self.run_root_, ignore_errors=True)

    def __del__(self):
        self._cleanup_artifacts()
