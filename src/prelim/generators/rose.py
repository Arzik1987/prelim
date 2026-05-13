import numpy as np

from .base import BaseGenerator


class Gen_rose(BaseGenerator):
    def __init__(self, smoothing=0.1, balanced=False, clip=True, seed=2020):
        super().__init__("rose", seed=seed)
        if smoothing < 0:
            raise ValueError("smoothing must be non-negative")
        self.smoothing_ = smoothing
        self.balanced_ = balanced
        self.clip_ = clip
        self.classes_ = None
        self.priors_ = None
        self.rows_ = None
        self.scales_ = None
        self.minimum_ = None
        self.maximum_ = None

    def fit(self, X, y=None, metamodel=None):
        del metamodel
        if y is None:
            raise ValueError("Gen_rose.fit requires y")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.minimum_ = X.min(axis=0)
        self.maximum_ = X.max(axis=0)
        global_scale = np.maximum(X.std(axis=0), 1e-6)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts / counts.sum()
        self.rows_ = {}
        self.scales_ = {}

        for cls in self.classes_:
            rows = X[y == cls]
            scale = np.maximum(rows.std(axis=0), global_scale * 1e-6) * self.smoothing_
            self.rows_[cls] = rows
            self.scales_[cls] = scale

        return self

    def _class_counts(self, n_samples):
        if self.balanced_:
            base = n_samples // len(self.classes_)
            counts = np.full(len(self.classes_), base, dtype=int)
            remainder = n_samples - counts.sum()
            if remainder:
                counts[:remainder] += 1
            return counts

        return self.rng_.multinomial(n_samples, self.priors_)

    def sample(self, n_samples=1):
        if self.classes_ is None:
            raise RuntimeError("Gen_rose.sample called before fit")

        rows = []
        for cls, count in zip(self.classes_, self._class_counts(n_samples)):
            if count == 0:
                continue
            source = self.rows_[cls]
            indices = self.rng_.choice(np.arange(len(source)), size=count, replace=True)
            sampled = source[indices].copy()
            sampled += self.rng_.normal(scale=self.scales_[cls], size=sampled.shape)
            rows.append(sampled)

        X = np.vstack(rows)
        if self.clip_:
            X = np.clip(X, self.minimum_, self.maximum_)

        xdim = X.shape[0]
        return X[self.rng_.choice(np.arange(xdim), size=xdim, replace=False), :].copy()
