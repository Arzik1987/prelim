import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import _tree
# see https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree


class Gen_rfdens:
    
    def __init__(self, seed=2020):
        self.seed_ = seed
        self.boxes_ = None
        self.nsamples_ = None

    def _get_rules_tree(self, tree, box):
        tree_ = tree.tree_
        feature_id = [i if i != _tree.TREE_UNDEFINED else None for i in tree_.feature]
     
        def recurse(node, box, boxes, nsamples):
            
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                ind = feature_id[node]
                threshold = tree_.threshold[node]
                b1 = box.copy()
                b1[1, ind] = np.round(threshold, 3)
                recurse(tree_.children_left[node], b1, boxes, nsamples)
                b2 = box.copy()
                b2[0, ind] = np.round(threshold, 3)
                recurse(tree_.children_right[node], b2, boxes, nsamples)
            else:
                boxes.append(box)
                nsamples.append(tree_.n_node_samples[node])
        
        boxes = []
        nsamples = []
        recurse(0, box, boxes, nsamples)
            
        return boxes, nsamples

    def fit(self, X, y, metamodel=None):
        self.boxes_ = []
        self.nsamples_ = []
        params = {"max_features": [2, "sqrt", None]}
        cv_rf = GridSearchCV(RandomForestClassifier(random_state=self.seed_), params, cv=5)
        cv_rf.fit(X, y)
        model = cv_rf.best_estimator_
        box = np.vstack((X.min(axis=0), X.max(axis=0)))
    
        for i in model.estimators_:
            tmpb, tmpn = self._get_rules_tree(i, box)
            self.boxes_ = self.boxes_ + tmpb
            self.nsamples_ = self.nsamples_ + tmpn
            
        self.nsamples_ = np.array(self.nsamples_)

    def sample(self, n_samples=1):
        niter = int(np.ceil(n_samples/sum(self.nsamples_)))
        X = []
        
        for _ in range(0, niter):
            for i in range(0, len(self.nsamples_)):
                box = self.boxes_[i]
                sidelen = box[1, :] - box[0, :]
                X.append(np.random.random((self.nsamples_[i], len(sidelen)))*sidelen + box[0, :])
        
        X = np.concatenate(X)
        xdim = X.shape[0]
        X = X[np.random.RandomState(self.seed_).choice(np.arange(xdim), size=xdim, replace=False), :].copy()
        return X[0:n_samples, :]

    def my_name(self):
        return "cmmrf"
    

# =============================================================================
# # TEST 
# 
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 500)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
# x = x[((x <= [6,6]) & (x>=[-1,-1])).all(axis = 1)]
# y = np.ones(x.shape[0])
# y[(x <= [4,1]).all(axis = 1)] = 0
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1], c = y)
# 
# 
# rfdens = Gen_rfdens()
# rfdens.fit(x, y)
# df = rfdens.sample(n_samples = 10)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================


