import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

from .base import BaseGenerator


class Gen_gmm(BaseGenerator):
    def __init__(self, params: dict = None, cv=5, seed=2020):
        super().__init__("gmmcv", seed=seed)
        if params is None:
            self.params_ = {
                "covariance_type": ["full", "tied", "diag", "spherical"],
                "n_components": list(range(1, 30)),
            }
        else:
            self.params_ = params

        self.model_ = None
        self.cv_ = cv

    def fit(self, X, y=None, metamodel=None):
        self.model_ = GridSearchCV(
            GaussianMixture(random_state=self.seed_),
            self.params_,
            cv=self.cv_,
        ).fit(X).best_estimator_
        return self

    def sample(self, n_samples=1):
        return self.model_.sample(n_samples)[0]


class Gen_gmmbic(BaseGenerator):
    def __init__(self, params: dict = None, cv=None, seed=2020):
        super().__init__("gmm", seed=seed)
        if params is None:
            self.params_ = {
                "covariance_type": ["full", "tied", "diag", "spherical"],
                "n_components": list(range(1, 30)),
            }
        else:
            self.params_ = params
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        # see https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
        lowest_bic = np.inf
        best_gmm = None
        for cv_type in self.params_["covariance_type"]:
            for n_components in self.params_["n_components"]:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cv_type,
                    random_state=self.seed_,
                )
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm

        self.model_ = best_gmm
        return self

    def sample(self, n_samples=1):
        return self.model_.sample(n_samples)[0]


class Gen_gmmbical(BaseGenerator):
    def __init__(self, params: dict = None, cv=None, seed=2020):
        super().__init__("gmmal", seed=seed)
        if params is None:
            self.params_ = {"n_components": list(range(1, 30))}
        else:
            self.params_ = params

    def fit(self, X, y=None, metamodel=None):
        # see https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
        lowest_bic = np.inf
        best_gmm = None
        for n_components in self.params_["n_components"]:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="diag",
                random_state=self.seed_,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm

        self.model_ = best_gmm
        return self

    def sample(self, n_samples=1):
        return self.model_.sample(n_samples)[0]


class Gen_classgmm(BaseGenerator):
    def __init__(self, params: dict = None, balanced=False, seed=2020):
        super().__init__("class_gmm", seed=seed)
        if params is None:
            self.params_ = {
                "covariance_type": ["full", "tied", "diag", "spherical"],
                "n_components": list(range(1, 10)),
            }
        else:
            self.params_ = params
        self.balanced_ = balanced
        self.classes_ = None
        self.priors_ = None
        self.models_ = None
        self.singletons_ = None
        self.scale_ = None

    def fit(self, X, y=None, metamodel=None):
        del metamodel
        if y is None:
            raise ValueError("Gen_classgmm.fit requires y")

        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.priors_ = counts / counts.sum()
        self.models_ = {}
        self.singletons_ = {}
        self.scale_ = np.maximum(X.std(axis=0), 1e-6) * 1e-6

        for cls in self.classes_:
            Xcls = X[y == cls]
            if len(Xcls) == 1:
                self.singletons_[cls] = Xcls[0].copy()
                continue

            lowest_bic = np.inf
            best_gmm = None
            for cv_type in self.params_["covariance_type"]:
                for n_components in self.params_["n_components"]:
                    if n_components > len(Xcls):
                        continue
                    gmm = GaussianMixture(
                        n_components=n_components,
                        covariance_type=cv_type,
                        random_state=self.seed_,
                    )
                    gmm.fit(Xcls)
                    bic = gmm.bic(Xcls)
                    if bic < lowest_bic:
                        lowest_bic = bic
                        best_gmm = gmm

            self.models_[cls] = best_gmm

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
            raise RuntimeError("Gen_classgmm.sample called before fit")

        rows = []
        for cls, count in zip(self.classes_, self._class_counts(n_samples)):
            if count == 0:
                continue
            if cls in self.singletons_:
                center = np.tile(self.singletons_[cls], (count, 1))
                rows.append(center + self.rng_.normal(scale=self.scale_, size=center.shape))
            else:
                rows.append(self.models_[cls].sample(count)[0])

        X = np.vstack(rows)
        xdim = X.shape[0]
        return X[self.rng_.choice(np.arange(xdim), size=xdim, replace=False), :].copy()
