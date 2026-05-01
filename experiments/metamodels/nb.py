import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from .base import BaseMetaModel


class Meta_nb(BaseMetaModel):
    def __init__(self):
        super().__init__("nb")

    def _fit_impl(self, X, y):
        model = CalibratedClassifierCV(estimator=GaussianNB())
        model.fit(X, y)
        cvscore = np.nanmean(cross_val_score(model, X, y))
        return model, cvscore
