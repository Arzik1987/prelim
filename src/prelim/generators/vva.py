from .vva_base import BaseVVA


class Gen_vva(BaseVVA):

    def _decision_scores(self, X, metamodel):
        return metamodel.predict_proba(X) - 0.5
