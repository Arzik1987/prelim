from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from .base import BaseMetaModel


class Meta_svm(BaseMetaModel):
    def __init__(self, params=None, cv=5, seed=2020):
        if params is None:
            params = {
                "C": [0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1],
            }
        super().__init__("svm", seed=seed)
        self.params_ = params
        self.cv_ = cv

    def _fit_impl(self, X, y):
        model = CalibratedClassifierCV(
            estimator=GridSearchCV(SVC(random_state=self.seed_), self.params_, cv=self.cv_)
        )
        model.fit(X, y)
        return model
