import numpy as np

from .base import BaseGenerator

try:
    from ForestDiffusion import ForestDiffusionModel
except Exception:
    ForestDiffusionModel = None


class Gen_forestdiffusion(BaseGenerator):
    def __init__(self, model_kwargs: dict | None = None, seed=2020):
        super().__init__("forestdiffusion", seed=seed)
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.X_ = None
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = np.asarray(X).copy()
        model_class = ForestDiffusionModel
        if model_class is None:
            from ForestDiffusion import ForestDiffusionModel as model_class

        self.model_ = model_class(self.X_, seed=self.seed_, **self.model_kwargs_)
        return self

    def sample(self, n_samples=1):
        sampled = self.model_.generate(batch_size=n_samples)
        return np.asarray(sampled)
