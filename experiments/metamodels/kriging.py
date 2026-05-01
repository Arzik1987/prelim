from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from .base import BaseMetaModel


class Meta_kriging(BaseMetaModel):
    def __init__(self, seed=2020):
        super().__init__("kriging", seed=seed)

    def _fit_impl(self, X, y):
        model = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=self.seed_)
        model.fit(X, y)
        return model
