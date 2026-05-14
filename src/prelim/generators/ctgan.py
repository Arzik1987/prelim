import numpy as np
import pandas as pd
from ctgan import CTGAN

from .base import BaseGenerator


class Gen_ctgan(BaseGenerator):
    def __init__(self, model_kwargs: dict | None = None, seed=2020):
        super().__init__("ctgan", seed=seed)
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.X_ = None
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = np.asarray(X).copy()
        self.model_ = CTGAN(**self.model_kwargs_)
        columns = [str(index) for index in range(self.X_.shape[1])]
        self.model_.fit(pd.DataFrame(self.X_, columns=columns), discrete_columns=[])
        return self

    def sample(self, n_samples=1):
        sampled = self.model_.sample(n_samples)
        return sampled.to_numpy()
