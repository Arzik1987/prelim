from sklearn.tree import _tree
import copy
import numpy as np


def opt_param(cvres, nval):
    # finds best hyperparameter. ignores NaN values occuring during cross-validation
    fit_res = np.empty((0, nval))
    for key, value in cvres.items():
        if 'split' in key:
            fit_res = np.vstack((fit_res, value))
    tmp = np.nanmean(fit_res, 0)
    return tmp


def n_leaves(tree):
    # counts the number of leaves in the trained decision tree
    # see https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    tree = copy.deepcopy(tree)
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]
    leaves = [(ls[i] == rs[i]) for i in nodes]
    n = np.sum([1 if i == _tree.TREE_UNDEFINED else 0 for i in dat.feature])

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
            n = n - 1   
    return n


def get_bi_param(nval, nattr):
    # determines a set of numbers (numbers of dimensions to restrict) for BI HPO procedure
    nattr = min(15, nattr)
    a = [ -x for x in range(-nattr, 0, np.ceil(nattr/nval).astype(int))]
    b = [ -x for x in range(-nattr, min(-nattr + nval, 0), 1)]
    res = a if len(a) > nval/2 + 1 else b
    return np.flip(res)


def get_new_test(Xtest, ytest, dsize, new_size = 10000):
    # cuts part of test data to be used as 'unlabelled pool' for semi-supervised tests
    n = min(int(np.floor(len(ytest)/2)), new_size - dsize)
    ytest = ytest[n:]
    Xnew = Xtest[:n,:].copy()
    Xtest = Xtest[n:,:]
    return Xtest, ytest, Xnew 

