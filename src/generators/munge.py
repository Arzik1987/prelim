
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Gen_munge:
    
    def __init__(self, local_var = 5, p_swap = 0.5, seed = 2020):
        if p_swap < 0.01:
            sys.exit("p_swap parameter is too small")
        self.p_swap_ = p_swap
        self.local_var_ = local_var
        self.seed_ = seed

    def fit(self, X):
        self.data_ = X.copy()
        self.index_ = NearestNeighbors(n_neighbors = 1).fit(self.data_).kneighbors()[1]
        return self
    
    def sample_once_(self):
        dtemp = self.data_.copy()
        for j in range(0, dtemp.shape[0]):
            nn = self.data_[self.index_[j],:].flatten()
            for k in range(0,dtemp.shape[1]):
                swap = np.random.uniform(0, 1, 1)
                if swap <= self.p_swap_:
                    dtemp[j, k] = np.random.normal(nn[k], abs(nn[k] - dtemp[j, k])/self.local_var_, 1)
        
        return dtemp

    def sample(self, n_samples):
        reps = int(n_samples/(self.data_.shape[0]*(1 - (1 - self.p_swap_)**self.data_.shape[1])) + 1)
        dlist = []

        for i in range(0, reps):
            dlist.append(self.sample_once_())
        new_data = np.unique(np.concatenate(dlist, axis = 0), axis = 0)
        
        # if the number of generated points is still lower than the required, generate more
        while new_data.shape[0] < n_samples:
            new_data = np.unique(np.concatenate([new_data, self.sample_once_()], axis = 0), axis = 0)
        inds = np.random.RandomState(self.seed_).choice(np.arange(new_data.shape[0]), size = new_data.shape[0], replace = False)
        
        return new_data[inds,:][:n_samples,:]



# =============================================================================
# # TEST 
# # TODO^ compare to R
# 
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 50)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 50)))
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1])
# 
# munge = Gen_munge(local_var = 1)
# munge.fit(x)
# df = munge.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================




# =============================================================================
# class Gen_munge:
#     
#     def __init__(self, local_var = 5, p_swap = 0.5):
#         if p_swap < 0.01:
#             sys.exit("p_swap parameter is too small")
#         self.p_swap_ = p_swap
#         self.local_var_ = local_var
# 
#     def fit(self, X):
#         self.data = X.copy()
#         self.index_ = NearestNeighbors(n_neighbors = 1).fit(self.data).kneighbors()[1]
#         return self
# 
#     def sample(self, n_samples):
#         index = NearestNeighbors(n_neighbors = 1).fit(self.data).kneighbors()[1]
#         reps = int(n_samples/(self.data.shape[0]*(1 - (1 - self.p_swap)**self.data.shape[1])) + 1)
#         dlist = []
# 
#         for i in range(0, reps):
#             dtemp = self.data.copy().to_numpy()
#             for j in range(0, dtemp.shape[0]):
#                 nn = self.data.to_numpy()[index[j],:].flatten()
#                 for k in range(0,dtemp.shape[1]):
#                     swap = np.random.uniform(0, 1, 1)
#                     if swap <= self.p_swap:
#                         dtemp[j, k] = np.random.normal(nn[k], abs(nn[k] - dtemp[j, k])/self.local_var, 1)
#             dlist.append(dtemp)
# 
#         new_data = np.concatenate(dlist, axis = 0)
#         sort_ind = sorted(np.unique(new_data, axis = 0, return_index = True)[1])
#         new_data = new_data[sort_ind,:]
#         
#         # if the number of generated points is still lower than the required, generate more
#         while new_data.shape[0] < n_samples:
#             dtemp = self.data.to_numpy()
#             for j in range(0, dtemp.shape[0]):
#                 nn = self.data.to_numpy()[index[j,:]].flatten()
#                 for k in range(0,dtemp.shape[1]):
#                     swap = np.random.uniform(0, 1, 1)
#                     if swap <= self.p_swap:
#                         dtemp[j, k] = np.random.normal(nn[k], abs(nn[k] - dtemp[j, k])/self.local_var, 1)
#             new_data = np.concatenate([new_data, dtemp], axis = 0)
#             sort_ind = sorted(np.unique(new_data, axis = 0, return_index = True)[1])
#             new_data = new_data[sort_ind,:]
#         
#         return pd.DataFrame(new_data[:n_samples,:], columns = self.cnames)
# 
# =============================================================================

# =============================================================================
# TODO: maybe also compare to R implementation?
#
# df = pd.read_csv("testdata.csv")
# df = df.iloc[:,[0,1]]
# x = Gen_munge(p_swap = 0.5, local_var = 0.5)
# x.fit(df)
# df1 = x.sample(n_samples = 201)
# 
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(df.iloc[:,0], df.iloc[:,1], s=10, c='b', marker="s", label='original')
# ax1.scatter(df1.iloc[:,0], df1.iloc[:,1], s=10, c='r', marker="o", label='generated')
# plt.legend(loc='upper right');
# plt.show()
# =============================================================================

'''
import math

import annoy as ann
import numpy as np
import pandas as pd

numeric = [int, float, np.int64, np.float, np.float64]

# Preferred method because its way faster than using exact NNs.


class MUNGE:

    def __init__(self, p=0.5, s=1, approximate=False):
        self.data = None
        self.s = s
        self.p = p
        self.approximate = approximate

    def fit(self, X: pd.DataFrame, **kwargs):
        self.data = X
        return self

    def sample(self, size: int):
        k = self.determine_k(size)
        if self.approximate:
            g_data = self.approximate_munge_annoy(self.data, k, self.p, self.s)
        else:
            g_data = self.munge_annoy(self.data, k, self.p, self.s)
        return g_data.sample(size)

    """
    MUNGE doubles size of dataset with each iteration. For a given dataset with size n_original and a demanden sample of
    size n_samples find k so that:
        n_original * 2^k >= n_samples <=> k >= log2(n_samples) - log2(n_original)
        
        If data doubles each iteration
        
        math.ceil(np.log2(sample_size) - np.log2(self.data.shape[0]))
    """
    def determine_k(self, sample_size):
        if sample_size < self.data.shape[0]:
            return 0
        return math.ceil(sample_size / self.data.shape[0])

    def munge_annoy(self, df: pd.DataFrame, k, p: float = 0.5, s: float = 1) -> pd.DataFrame:
        result = df
        nn_idx: ann.AnnoyIndex = self.__nn_index(df)
        for i in range(k):
            data = df.copy(deep=True)
            data = data.reset_index(drop=True)
            for e_idx, value in data.iterrows():
                nn = nn_idx.get_nns_by_item(e_idx, 2).pop()
                for attr in data:
                    if np.random.random() > p:
                        if type(data[attr][e_idx]) in numeric:
                            sd = np.abs(data.loc[e_idx, attr] - data.loc[nn, attr]) / s
                            nn_sample = np.random.normal(data.loc[e_idx, attr], sd, 1)
                            e_sample = np.random.normal(data.loc[nn, attr], sd, 1)
                            data.loc[e_idx, attr] = e_sample
                            data.loc[nn, attr] = nn_sample
                        else:
                            old_element = data.loc[e_idx, attr]
                            element = data.loc[nn, attr]
                            data.loc[e_idx, attr] = element
                            data.loc[nn, attr] = old_element
                            data = data.reset_index(drop=True) # TODO review if this is correct
            result = result.reset_index(drop=True)
            result = result.append(data)
        return result

    def approximate_munge_annoy(self, df: pd.DataFrame, k, p: float = 0.5, s: float = 1) -> pd.DataFrame:
        result = pd.DataFrame(columns=df.columns)
        nn_idx: ann.AnnoyIndex = self.__nn_index(df)
        orig_data = df.reset_index(drop=True)
        for e_idx in range(orig_data.shape[0] - 1):
            nn = nn_idx.get_nns_by_item(e_idx, 2).pop()
            data = pd.DataFrame()
            for attr in result.columns:
                sd = np.abs(orig_data.loc[e_idx, attr] - orig_data.loc[nn, attr]) / s
                nn_sample = np.random.normal(orig_data.loc[e_idx, attr], sd, round(k * p))
                e_sample = np.random.normal(orig_data.loc[nn, attr], sd, round(k * p))
                data.insert(loc=0, column=attr, value=e_sample + nn_sample)
                data = data.reset_index(drop=True)  # TODO review if this is correct
                result = result.append(data)
                result = result.reset_index(drop=True)
        return result

    def __nn_index(self, data: pd.DataFrame, n_trees: int = 10) -> ann.AnnoyIndex:
        t = ann.AnnoyIndex(data.shape[1], 'euclidean')  # Length of item vector that will be indexed
        for row_idx in range(data.shape[0]):
            t.add_item(row_idx, data.iloc[row_idx, :])
        t.build(n_trees)
        return t
'''