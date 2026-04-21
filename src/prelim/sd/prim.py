import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted


class PRIM(BaseEstimator):
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.box_ = self._get_initial_restrictions(X)
        self.N_ = len(y)
        self.Np_ = np.sum(y)
        self.mult_ = self.N_ ** 2 / (self.N_ - self.Np_)

        highest = self._target_fun(self.Np_, self.N_)
        cont = True
        box = self.box_.copy()
        i = 1
        while i < 100 and cont:
            hgh, cont = self._peel_one()
            if hgh > highest:
                highest = hgh
                box = self.box_.copy()
            if np.sum(self.y_) < highest * self.mult_:
                cont = False
            i += 1

        self.X_ = None
        self.y_ = None
        self.box_ = box
        self.N_ = None
        self.Np_ = None

        return self

    def _peel_one(self):
        hgh, bnd = -np.inf, -np.inf
        rn, cn = -1, -1
        cont = False
        for i in range(self.X_.shape[1]):
            if len(np.unique(self.X_[:, i])) > 1:
                cont = True
                bound = np.quantile(self.X_[:, i], self.alpha, method="midpoint")
                retain = self.X_[:, i] > bound
                if np.count_nonzero(retain) == 0:
                    retain = self.X_[:, i] >= bound
                tar = self._target_fun(np.sum(self.y_[retain]), np.count_nonzero(retain))
                if tar > hgh:
                    hgh = tar
                    inds = retain
                    rn = 0
                    cn = i
                    bnd = bound
                bound = np.quantile(self.X_[:, i], 1 - self.alpha, method="midpoint")
                retain = self.X_[:, i] < bound
                if np.count_nonzero(retain) == 0:
                    retain = self.X_[:, i] <= bound
                tar = self._target_fun(np.sum(self.y_[retain]), np.count_nonzero(retain))
                if tar > hgh:
                    hgh = tar
                    inds = retain
                    rn = 1
                    cn = i
                    bnd = bound

        if cont:
            self.X_ = self.X_[inds]
            self.y_ = self.y_[inds]
            self.box_[rn, cn] = bnd

        return hgh, cont

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y):
        X, y = check_X_y(X, y)
        check_is_fitted(self)
        box = self.box_
        ind_in_box = np.ones(len(y), dtype=bool)
        for i in range(box.shape[1]):
            ind_in_box = np.logical_and(
                ind_in_box,
                np.logical_and(X[:, i] >= box[0, i], X[:, i] <= box[1, i]),
            )

        if np.sum(ind_in_box) == 0:
            return np.nan
        return (np.sum(ind_in_box) / len(y)) * (
            np.sum(y[ind_in_box]) / np.sum(ind_in_box) - np.sum(y) / len(y)
        )

    def _get_initial_restrictions(self, X):
        return np.vstack((np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf)))

    def _target_fun(self, npos, n):
        return (n / self.N_) * (npos / n - self.Np_ / self.N_)

    def get_nrestr(self):
        return np.count_nonzero(
            np.any(
                np.all([[self.box_ != np.inf], [self.box_ != -np.inf]], axis=0),
                axis=1,
            )
        )
