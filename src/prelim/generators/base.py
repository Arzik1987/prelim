from abc import ABC

import numpy as np


class BaseGenerator(ABC):

    def __init__(self, name, seed=2020):
        self.name_ = name
        self.seed_ = seed
        self.rng_ = np.random.RandomState(seed)

    def my_name(self):
        return self.name_
