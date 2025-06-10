import numpy as np


class Gen_noise:

    def __init__(self, scale=0.3):
        self.scale_ = scale

    def fit(self, X, y=None, metamodel=None):
        self.data_ = X.copy()
        self.data_ = self.data_.astype(float)
        return self

    def sample(self, n_samples=1):
        if self.data_ is None:
            raise RuntimeError("Generator must be fitted before sampling")

        # randomly choose data points to perturb so that the output size matches
        # the requested number of samples
        mod_data = self.data_[
            np.random.choice(self.data_.shape[0], n_samples, replace=True)
        ].astype(float)

        for col in range(mod_data.shape[1]):
            unique_vals = np.unique(self.data_[:, col])
            if len(unique_vals) > 1:
                step = np.min(np.diff(np.sort(unique_vals))) * self.scale_
            else:
                step = self.scale_
            mod_data[:, col] += np.random.uniform(-step, step, n_samples)

        return mod_data
    
    def my_name(self):
        return "noise"
    

# =============================================================================
# # TEST
# 
# x = np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]], dtype=float)
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1])
# 
# ng = Gen_noise()
# ng.fit(x)
# df = ng.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================