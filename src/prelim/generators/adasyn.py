import numpy as np
from imblearn.over_sampling import ADASYN
import warnings
from .rand import Gen_randu


class Gen_adasyn:

    def __init__(self):
        self.X_ = None
        self.mname_ = "adasyn"

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        return self

    def sample(self, n_samples=1):
        parss = 'not majority'
        if self.X_.shape[0] > n_samples:
            warnings.warn("The required sample size is smaller than the number of observations in train")
            parss = 'all'

        y = np.ones(self.X_.shape[0]), np.zeros(n_samples)
        y = np.concatenate(y)

        X = np.concatenate((self.X_, Gen_randu().fit(self.X_).sample(n_samples = n_samples)))
        Xnew = None
        parknn = min(5, n_samples, self.X_.shape[0])

        # TODO Inspect
        while type(Xnew) is not np.ndarray and parknn <= n_samples and parknn <= self.X_.shape[0]:
            try:
                Xnew, y = ADASYN(sampling_strategy = parss, n_neighbors = parknn, random_state = 2020).fit_resample(X, y)
            except (ValueError, RuntimeError):
                parknn = parknn * 2

        if type(Xnew) is not np.ndarray:
            from imblearn.over_sampling import SMOTE
            parknn = min(5, n_samples, self.X_.shape[0])
            Xnew, y = SMOTE(sampling_strategy = parss, k_neighbors = parknn, random_state = 2020).fit_resample(X, y)
            self.mname_ = "adasyns"
        else:
            self.mname_ = "adasyn"
        
        return Xnew[y == 1,:][0:n_samples,:]
    
    def my_name(self):
        return self.mname_
    

# =============================================================================
# # TEST
# 
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples = 100, n_features = 2, n_informative = 2,
#                            n_redundant = 0, n_repeated = 0, n_classes = 1, 
#                            random_state = 0)
# import matplotlib.pyplot as plt
# plt.scatter(X[:,0], X[:,1])
# 
# ada_gen = Gen_adasyn()
# ada_gen.fit(X)
# df = ada_gen.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================

