import numpy as np


class DummyMeta:

    def __init__(self, *args):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(len(X) * [1])
