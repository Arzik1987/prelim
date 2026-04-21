import numpy as np

from .vva_base import BaseVVA


class Gen_vva(BaseVVA):

    def _decision_scores(self, X, metamodel):
        class_one_ind = np.where(metamodel.classes_ == 1)[0][0]
        return metamodel.predict_proba(X)[:, class_one_ind] - 0.5
