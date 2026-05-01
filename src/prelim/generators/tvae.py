import numpy as np
import pandas as pd
from ctgan import TVAE

from .base import BaseGenerator


class Gen_tvae(BaseGenerator):
    def __init__(self, model_kwargs: dict | None = None, seed=2020):
        super().__init__("tvae", seed=seed)
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.X_ = None
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = np.asarray(X).copy()
        self.model_ = TVAE(**self.model_kwargs_)
        self.model_.fit(pd.DataFrame(self.X_), discrete_columns=[])
        return self

    def sample(self, n_samples=1):
        sampled = self.model_.sample(n_samples)
        return sampled.to_numpy()
