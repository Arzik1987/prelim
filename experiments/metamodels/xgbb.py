import numpy as np
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .base import BaseMetaModel

    
class Meta_xgb_bal(BaseMetaModel):
    def __init__(self, params=None, cv=5, seed=2020):
        if params is None:
            params = {
                "n_estimators": randint(10, 990),
                "learning_rate": uniform(0.0001, 0.2),
                "gamma": uniform(0, 0.4),
                "max_depth": [6],
                "subsample": uniform(0.5, 0.5),
            }
        super().__init__("xgbb", seed=seed)
        self.params_ = params
        self.cv_ = cv

    def _fit_impl(self, X, y):
        nz = np.count_nonzero(y)
        spv = (y.shape[0] - nz) / nz
        search = RandomizedSearchCV(
            XGBClassifier(
                nthread=1,
                verbosity=0,
                use_label_encoder=False,
                scale_pos_weight=spv,
            ),
            self.params_,
            random_state=self.seed_,
            cv=self.cv_,
            n_iter=50,
            n_jobs=1,
            scoring="balanced_accuracy",
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_score_
