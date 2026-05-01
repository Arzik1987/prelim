from abc import ABC, abstractmethod

import numpy as np


class BaseMetaModel(ABC):
    def __init__(self, name, seed=2020):
        self.name_ = name
        self.seed_ = seed
        self.model_ = None
        self.cvscore_ = None

    def fit(self, X, y):
        result = self._fit_impl(X, y)
        if isinstance(result, tuple):
            self.model_, self.cvscore_ = result
        else:
            self.model_ = result
            self.cvscore_ = None
        return self

    @abstractmethod
    def _fit_impl(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        index = int(np.flatnonzero(self.model_.classes_ == 1)[0])
        return self.model_.predict_proba(X)[:, index]

    def fit_score(self):
        return self.cvscore_

    def my_name(self):
        return self.name_
