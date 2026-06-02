import numpy as np

from .vva_base import BaseVVA


class Gen_vva(BaseVVA):

    def _decision_scores(self, X, metamodel):
        scores = np.asarray(metamodel.predict_proba(X))
        if scores.ndim == 1:
            return scores - 0.5
        if scores.ndim != 2:
            raise ValueError("Gen_vva requires predict_proba(...) to return a 1D or 2D array")
        if scores.shape[1] == 1:
            return scores[:, 0] - 0.5

        classes = getattr(metamodel, "classes_", None)
        if classes is not None:
            matches = np.where(np.asarray(classes) == 1)[0]
            if len(matches) == 1:
                return scores[:, int(matches[0])] - 0.5

        if scores.shape[1] == 2:
            return scores[:, 1] - 0.5

        raise ValueError("Gen_vva requires binary predict_proba output or classes_ containing label 1")
