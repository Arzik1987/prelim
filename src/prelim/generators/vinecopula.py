import numpy as np
import pandas as pd

from .base import BaseGenerator

try:
    from copulas.multivariate import VineCopula
except Exception:
    VineCopula = None


class Gen_vinecopula(BaseGenerator):
    def __init__(self, vine_type="center", model_kwargs: dict | None = None, seed=2020):
        super().__init__("vinecopula", seed=seed)
        self.vine_type_ = vine_type
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.X_ = None
        self.columns_ = None
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        del y, metamodel
        self.X_ = np.asarray(X).copy()
        self.columns_ = [f"x{i}" for i in range(self.X_.shape[1])]
        train_df = pd.DataFrame(self.X_, columns=self.columns_)

        model_class = VineCopula
        if model_class is None:
            from copulas.multivariate import VineCopula as model_class

        self.model_ = model_class(self.vine_type_, random_state=self.seed_, **self.model_kwargs_)
        self.model_.fit(train_df)
        return self

    def sample(self, n_samples=1):
        if self.model_ is None:
            raise RuntimeError("Gen_vinecopula.sample called before fit")

        sampled = self.model_.sample(n_samples)
        return sampled.loc[:, self.columns_].to_numpy()
