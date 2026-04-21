from itertools import cycle

import numpy as np

from .base import BaseGenerator


class BaseVVA(BaseGenerator):

    def __init__(self, rho=0.2, seed=2020):
        super().__init__("vva", seed=seed)
        self.rho_ = rho
        self.generate_ = None
        self.nn_ = None
        self.ordinds_ = None
        self.Xbound_ = None
        self.dim_ = None
        self.trainn_ = None

    def fit(self, X, metamodel, y=None):
        self.generate_ = True
        self.dim_ = X.shape[1]
        self.trainn_ = X.shape[0]
        X_train = X.copy()
        scores = self._decision_scores(X_train, metamodel)
        if sum(scores < 0) == 0 or sum(scores > 0) == 0:
            self.generate_ = False
            return self

        inds = np.concatenate(
            (
                np.where(scores == max(scores[scores < 0]))[0],
                np.where(scores == min(scores[scores > 0]))[0],
            )
        )
        X_bound = X_train[inds, :].copy()
        y_bound = scores[inds].copy()
        X_train = np.delete(X_train, inds, axis=0)
        scores = np.delete(scores, inds)
        n_rest = int(np.ceil(X.shape[0] * self.rho_ - len(inds)))
        if n_rest > 0:
            inds = np.argsort(abs(scores))[:n_rest]
            X_bound = np.concatenate((X_bound, X_train[inds, :].copy()), axis=0)
            y_bound = np.concatenate((y_bound, scores[inds].copy()))

        self._find_neighbours(X_bound, y_bound)
        return self

    def _decision_scores(self, X, metamodel):
        raise NotImplementedError

    def _find_neighbours(self, X, y):
        X_pos = X[y > 0, :]
        X_neg = X[y < 0, :]
        nn_pos, dist_pos = self._nearest_neighbours(X_pos, X_neg)
        nn_neg, dist_neg = self._nearest_neighbours(X_neg, X_pos)
        self.Xbound_ = np.concatenate((X_pos, X_neg), axis=0)
        self.nn_ = np.concatenate((nn_pos + len(nn_pos), nn_neg))
        self.ordinds_ = np.argsort(np.concatenate((dist_pos, dist_neg)))

    def _nearest_neighbours(self, X1, X2):
        X1, X2 = map(np.asarray, (X1, X2))
        nearest_neighbour = np.empty((len(X1),), dtype=np.intp)
        dist = np.empty((len(X1),), dtype=np.float32)
        for j, xj in enumerate(X1):
            idx = np.argmin(np.sum((X2 - xj) ** 2, axis=1))
            nearest_neighbour[j] = idx
            dist[j] = np.sqrt(np.sum((X2[idx] - xj) ** 2))

        return nearest_neighbour, dist

    def sample(self, r):
        if r < 0 or r > 2.5:
            raise ValueError("the boundaries for r defined in the paper are from 0 to 2.5")
        if r == 0 or self.generate_ is False:
            return np.empty((0, self.dim_))

        n_generated = int(np.ceil(self.trainn_ * r))
        thetas = self.rng_.uniform(0, 1, (n_generated, self.dim_))
        pool = cycle(self.ordinds_)
        new_points = []
        for k in range(n_generated):
            bound_index = next(pool)
            nn_index = self.nn_[bound_index]
            theta = thetas[k, :]
            new_points.append(
                self.Xbound_[bound_index, :] * theta
                + self.Xbound_[nn_index, :] * (1 - theta)
            )

        return np.vstack(new_points)

    def will_generate(self):
        return self.generate_
