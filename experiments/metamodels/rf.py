from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from .base import BaseMetaModel


class Meta_rf(BaseMetaModel):
    def __init__(self, params=None, cv=5, seed=2020):
        if params is None:
            params = {"max_features": [2, "sqrt", None]}
        super().__init__("rf", seed=seed)
        self.params_ = params
        self.cv_ = cv

    def _fit_impl(self, X, y):
        search = GridSearchCV(
            RandomForestClassifier(random_state=self.seed_),
            self.params_,
            cv=self.cv_,
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_score_
