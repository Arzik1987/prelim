import os

import numpy as np
import pandas as pd

from .base import BaseGenerator

try:
    from be_great import GReaT
except Exception:
    GReaT = None


class Gen_great(BaseGenerator):
    def __init__(self, model_kwargs: dict | None = None, sample_kwargs: dict | None = None, seed=2020):
        super().__init__("great", seed=seed)
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.model_kwargs_.setdefault("llm", os.environ.get("PRELIM_GREAT_LLM", "distilgpt2"))
        self.sample_kwargs_ = {} if sample_kwargs is None else dict(sample_kwargs)
        self.sample_kwargs_.setdefault("device", os.environ.get("PRELIM_GREAT_SAMPLE_DEVICE", "cpu"))
        self.sample_kwargs_.setdefault("max_length", int(os.environ.get("PRELIM_GREAT_SAMPLE_MAX_LENGTH", "32")))
        self.sample_kwargs_.setdefault("guided_sampling", True)
        self.sample_kwargs_.setdefault("random_feature_order", False)
        self.X_ = None
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        del y, metamodel
        self.X_ = np.asarray(X).copy()
        model_class = GReaT
        if model_class is None:
            from be_great import GReaT as model_class

        self.model_ = model_class(**self.model_kwargs_)
        train_df = pd.DataFrame(self.X_)
        train_df.columns = train_df.columns.map(str)
        self.model_.fit(train_df)
        return self

    def sample(self, n_samples=1):
        sampled = self.model_.sample(n_samples, **self.sample_kwargs_)
        return sampled.to_numpy()
