import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from src.generators.rand import Gen_randu


class Gen_smote:

    def __init__(self):
        self.X_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        return self

    def sample(self, n_samples=1):
        parss = 'not majority'
        if self.X_.shape[0] > n_samples:
            warnings.warn("The required sample size is smaller that the number of observations in train")
            parss = 'all'
        parknn = min(5, n_samples, self.X_.shape[0])
        y = np.concatenate((np.ones(self.X_.shape[0]), np.zeros(n_samples)))
        X = np.concatenate((self.X_, Gen_randu().fit(self.X_).sample(n_samples=n_samples)))
        X, y = SMOTE(sampling_strategy=parss, k_neighbors=parknn, random_state=2020).fit_resample(X, y)
        return X[y == 1, :][0:n_samples, :]
    
    def my_name(self):
        return "smote"
    

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
# smote_gen = Gen_smote()
# smote_gen.fit(X)
# df = smote_gen.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================
