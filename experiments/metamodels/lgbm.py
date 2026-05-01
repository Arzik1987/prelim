from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

from .base import BaseMetaModel


class Meta_lgbm(BaseMetaModel):
    def __init__(self, params=None, cv=5, seed=2020):
        if params is None:
            params = {
                "n_estimators": randint(10, 990),
                "learning_rate": uniform(0.0001, 0.2),
                "max_depth": [-1, 3, 5, 7, 9],
                "num_leaves": randint(15, 63),
                "subsample": uniform(0.5, 0.5),
            }
        super().__init__("lgbm", seed=seed)
        self.params_ = params
        self.cv_ = cv

    def _fit_impl(self, X, y):
        search = RandomizedSearchCV(
            LGBMClassifier(random_state=self.seed_, verbosity=-1),
            self.params_,
            random_state=self.seed_,
            cv=self.cv_,
            n_iter=50,
            n_jobs=1,
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_score_
