import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted


class BI(BaseEstimator):

    def __init__(self, depth=5, beam_size=1, add_iter=50):
        self.beam_size = beam_size
        self.depth = depth
        self.add_iter = add_iter

    def get_params(self, deep=True):
        return {
            "beam_size": self.beam_size,
            "depth": self.depth,
            "add_iter": self.add_iter,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.N_ = len(y)
        self.Np_ = np.sum(y)
        dim = X.shape[1]

        if np.logical_or(y.min() < 0, y.max() > 1):
            warnings.warn("The target variable takes values from outside [0,1]")
        if self.depth > dim:
            warnings.warn("Restricting depth parameter to the number of attributes in data")
        depth = min(self.depth, dim)

        box_init = self._get_initial_restrictions(X)
        res_box = []
        res_tab = np.empty([0, 3])

        for i in range(dim):
            box_new, quality, improved = self._refine(X, y, box_init, i, 0)
            res_box.append(box_new)
            res_tab = np.concatenate((res_tab, np.array([[quality, improved, i]])), axis=0)

        if depth > 1:
            add_iter = depth + self.add_iter
            while add_iter > 0:
                add_iter -= 1

                if res_tab.shape[0] > self.beam_size:
                    retain = res_tab[:, 0] >= np.sort(res_tab[:, 0])[::-1][self.beam_size - 1]
                    if np.sum(retain) < len(retain):
                        res_tab = res_tab[retain]
                        res_box = [res_box[i] for i in np.where(retain)[0]]
                    if len(res_box) > 1:
                        retain = self._get_dup_boxes(res_box)
                        if np.sum(retain) < len(retain):
                            res_tab = res_tab[retain]
                            res_box = [res_box[i] for i in np.where(retain)[0]]
                    if res_tab.shape[0] > self.beam_size:
                        sort_ind = res_tab[:, 0].argsort()[: self.beam_size]
                        res_tab = res_tab[sort_ind]
                        res_box = [res_box[i] for i in sort_ind]

                if res_tab[:, 1].sum() == 0:
                    add_iter = 0

                for k in range(len(res_tab)):
                    if res_tab[k, 1] == 1:
                        res_tab[k, 1] = 0
                        inds_r = np.where(np.equal(box_init, res_box[k]).sum(axis=0) < 2)[0]
                        if len(inds_r) < depth:
                            inds_r = np.arange(dim)
                        inds_r = inds_r[inds_r != res_tab[k, 2]]
                        for i in inds_r:
                            box_new, quality, improved = self._refine(X, y, res_box[k], i, res_tab[k, 0])
                            if improved == 1:
                                res_box.append(box_new)
                                res_tab = np.concatenate(
                                    (res_tab, np.array([[quality, improved, i]])),
                                    axis=0,
                                )

        winner = np.where(res_tab[:, 0] == max(res_tab[:, 0]))[0][0]
        self.box_ = res_box[winner]
        return self

    def score(self, X, y):
        X, y = check_X_y(X, y)
        check_is_fitted(self)

        ind_in_box = np.ones(len(y), dtype=bool)
        for i in range(self.box_.shape[1]):
            ind_in_box = np.logical_and(
                ind_in_box,
                np.logical_and(X[:, i] >= self.box_[0, i], X[:, i] <= self.box_[1, i]),
            )

        if np.sum(ind_in_box) == 0:
            return np.nan
        return (np.sum(ind_in_box) / len(y)) * (
            np.sum(y[ind_in_box]) / np.sum(ind_in_box) - np.sum(y) / len(y)
        )

    def _get_dup_boxes(self, boxes):
        inds = np.ones(len(boxes), dtype=bool)
        for i in range(len(boxes) - 1):
            for j in range(i + 1, len(boxes)):
                if inds[j] and np.array_equal(boxes[i], boxes[j]):
                    inds[j] = False
        return inds

    def _refine(self, X, y, box, ind, start_q):
        # The numbered comments below refer to Algorithm 3 in:
        # "Efficient algorithms for finding richer subgroup descriptions
        # in numeric and nominal data".
        ind_in_box = np.ones(self.N_, dtype=bool)
        for i in range(X.shape[1]):
            if i != ind:
                ind_in_box = np.logical_and(
                    ind_in_box,
                    np.logical_and(X[:, i] >= box[0, i], X[:, i] <= box[1, i]),
                )
        in_box = np.vstack((X[ind_in_box, ind], y[ind_in_box])).T
        in_box = in_box[in_box[:, 1].argsort()]

        t_m, h_m = -np.inf, -np.inf  # 3-4
        l, r = box[0, ind], box[1, ind]  # 1
        n = in_box.shape[0]
        npos = in_box[:, 1].sum()
        wracc_m = start_q  # 2

        t = np.unique(in_box[:, 0])  # define T; sorting happens implicitly
        itert = len(t)
        for i in range(itert):  # 5
            if i != 0:
                tmp = in_box[in_box[:, 0] == t[i - 1]]
                n = n - tmp.shape[0]  # 6
                npos = npos - tmp[:, 1].sum()  # 6
            h = self._wracc(n, npos, self.N_, self.Np_)  # 7
            if h > h_m:  # 8
                h_m = h  # 9
                if i == 0:  # 10
                    t_m = -np.inf
                else:
                    t_m = (t[i] + t[i - 1]) / 2

            tmp = in_box[np.logical_and(in_box[:, 0] >= t_m, in_box[:, 0] <= t[i])]
            n_i = tmp.shape[0]
            npos_i = tmp[:, 1].sum()
            wracc_i = self._wracc(n_i, npos_i, self.N_, self.Np_)
            if wracc_i > wracc_m:  # 11
                l = t_m  # 12
                if i == (itert - 1):  # 12
                    r = np.inf
                else:
                    r = (t[i] + t[i + 1]) / 2
                wracc_m = wracc_i  # 13
        box_new = box.copy()
        box_new[:, ind] = [l, r]
        return box_new, wracc_m, int(wracc_m != start_q)

    def _wracc(self, n, npos, N, Np):
        return (n / N) * (npos / n - Np / N)

    def _get_initial_restrictions(self, X):
        return np.vstack((np.full(X.shape[1], -np.inf), np.full(X.shape[1], np.inf)))

    def get_nrestr(self):
        return np.count_nonzero(
            np.any(
                np.all([[self.box_ != np.inf], [self.box_ != -np.inf]], axis=0),
                axis=1,
            )
        )
