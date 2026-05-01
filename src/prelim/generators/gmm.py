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
