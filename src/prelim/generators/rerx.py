from .base import BaseGenerator


class Gen_rerx(BaseGenerator):

    def __init__(self, rho=0.2, seed=2020):
        super().__init__("rerx", seed=seed)
        self.rho_ = rho
        self.X_ = None

    def fit(self, X, y=None, metamodel=None):
        if y is None or metamodel is None:
            raise ValueError("Gen_rerx.fit requires both y and metamodel")
        ypred = metamodel.predict(X)
        self.X_ = X[y == ypred]
        return self

    def sample(self, n_samples=1):
        return self.X_
