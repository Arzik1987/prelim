import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseGenerator


class Gen_kdeb(BaseGenerator):
    def __init__(self, knn=10, seed=2020):
        super().__init__("kdeb", seed=seed)
        self.knn_ = knn
        self.X_ = None
        self.dist_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        if self.knn_ == 0:
            self.dist_ = 1
        elif self.knn_ >= X.shape[0]:
            raise RuntimeError(
                "The dataset is too small or the knn value is too large. "
                "Number of data points must be greater than k."
            )
        else:
            self.dist_ = np.mean(NearestNeighbors(n_neighbors=self.knn_).fit(X).kneighbors()[0][:, self.knn_ - 1])
        return self

    def sample(self, n_samples):
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        d = self.X_.shape[1]
        u = self.rng_.normal(0, 1, (n_samples, d + 2))
        den = np.sum(u**2, axis=1) ** 0.5
        u = u / den[:, None]

        base_rows = self.rng_.choice(self.X_.shape[0], n_samples)
        return self.X_[base_rows, :] + u[:, 0:d]
