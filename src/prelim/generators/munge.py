import numpy as np
from sklearn.neighbors import NearestNeighbors
from .base import BaseGenerator


class Gen_munge(BaseGenerator):
    
    def __init__(self, local_var=5, p_swap=0.5, seed=2020):
        super().__init__("munge", seed=seed)
        if p_swap < 0.01:
            raise ValueError("p_swap parameter is too small")
        self.p_swap_ = p_swap
        self.local_var_ = local_var
        self.data_ = None
        self.index_ = None

    def fit(self, X, y=None, metamodel=None):
        self.data_ = X.copy()
        self.index_ = NearestNeighbors(n_neighbors=1).fit(self.data_).kneighbors()[1]
        return self
    
    def _sample_once(self):
        dtemp = self.data_.copy()
        for j in range(0, dtemp.shape[0]):
            nn = self.data_[self.index_[j], :].flatten()
            for k in range(0,dtemp.shape[1]):
                swap = self.rng_.uniform(0, 1, 1)
                if swap <= self.p_swap_:
                    dtemp[j, k] = self.rng_.normal(nn[k], abs(nn[k] - dtemp[j, k])/self.local_var_)
        
        return dtemp

    def sample(self, n_samples):
        reps = int(n_samples/(self.data_.shape[0]*(1 - (1 - self.p_swap_)**self.data_.shape[1])) + 1)
        dlist = []

        for i in range(0, reps):
            dlist.append(self._sample_once())
        new_data = np.unique(np.concatenate(dlist, axis=0), axis=0)
        
        # if the number of generated points is still lower than the required, generate more
        while new_data.shape[0] < n_samples:
            new_data = np.unique(np.concatenate([new_data, self._sample_once()], axis=0), axis=0)
        inds = self.rng_.choice(np.arange(new_data.shape[0]), size=new_data.shape[0], replace=False)
        
        return new_data[inds, :][:n_samples, :]
