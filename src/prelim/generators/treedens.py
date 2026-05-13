import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.tree import _tree

from .base import BaseGenerator


class Gen_treedens(BaseGenerator):
    def __init__(self, n_estimators=100, max_samples="auto", max_features=1.0, seed=2020):
        super().__init__("treedens", seed=seed)
        self.n_estimators_ = n_estimators
        self.max_samples_ = max_samples
        self.max_features_ = max_features
        self.model_ = None
        self.boxes_ = None
        self.nsamples_ = None

    def _get_rules_tree(self, tree, box, feature_map):
        tree_ = tree.tree_

        def recurse(node, box, boxes, nsamples):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                ind = feature_map[tree_.feature[node]]
                threshold = tree_.threshold[node]
                b1 = box.copy()
                b1[1, ind] = min(b1[1, ind], threshold)
                recurse(tree_.children_left[node], b1, boxes, nsamples)
                b2 = box.copy()
                b2[0, ind] = max(b2[0, ind], threshold)
                recurse(tree_.children_right[node], b2, boxes, nsamples)
            else:
                boxes.append(box)
                nsamples.append(tree_.n_node_samples[node])

        boxes = []
        nsamples = []
        recurse(0, box, boxes, nsamples)
        return boxes, nsamples

    def fit(self, X, y=None, metamodel=None):
        del y, metamodel
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.boxes_ = []
        self.nsamples_ = []
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators_,
            max_samples=self.max_samples_,
            max_features=self.max_features_,
            random_state=self.seed_,
        )
        self.model_.fit(X)
        box = np.vstack((X.min(axis=0), X.max(axis=0)))

        for estimator, features in zip(self.model_.estimators_, self.model_.estimators_features_):
            tmpb, tmpn = self._get_rules_tree(estimator, box, features)
            self.boxes_.extend(tmpb)
            self.nsamples_.extend(tmpn)

        self.nsamples_ = np.array(self.nsamples_, dtype=int)
        return self

    def sample(self, n_samples=1):
        if self.boxes_ is None or self.nsamples_ is None:
            raise RuntimeError("Gen_treedens.sample called before fit")

        total = int(sum(self.nsamples_))
        if total <= 0:
            raise RuntimeError("Gen_treedens has no covered leaf regions to sample from")

        niter = int(np.ceil(n_samples / total))
        X = []

        for _ in range(niter):
            for box, count in zip(self.boxes_, self.nsamples_):
                sidelen = box[1, :] - box[0, :]
                X.append(self.rng_.random_sample((count, len(sidelen))) * sidelen + box[0, :])

        X = np.concatenate(X)
        xdim = X.shape[0]
        X = X[self.rng_.choice(np.arange(xdim), size=xdim, replace=False), :].copy()
        return X[0:n_samples, :]
