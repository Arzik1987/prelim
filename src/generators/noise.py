import numpy as np

class Gen_noise:

    def __init__(self, scale = 0.3):
        self.scale_ = scale

    def fit(self, X):
        self.data_ = X.copy()
        self.data_ = self.data_.astype(float)
        return self

    def sample(self, n_samples = 1):
        mod_data = self.data_.copy()
        for col in range(0,mod_data.shape[1]):
            mindist = min(np.diff(np.unique(mod_data[:,col])))*self.scale_
            mod_data[:,col] = mod_data[:,col] + np.random.uniform(-mindist, mindist, len(mod_data[:,col]))
        return mod_data
    

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