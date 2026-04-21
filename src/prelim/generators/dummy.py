from .base import BaseGenerator


class Gen_dummy(BaseGenerator):
    def __init__(self, seed=2020):
        super().__init__("dummy", seed=seed)
        self.X_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        return self

    def sample(self, n_samples=1):
        return self.X_.copy()
