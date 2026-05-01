import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer

from .base import BaseGenerator


class Gen_copulagan(BaseGenerator):
    def __init__(self, model_kwargs: dict | None = None, seed=2020):
        super().__init__("copulagan", seed=seed)
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.X_ = None
        self.model_ = None
        self.metadata_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = np.asarray(X).copy()
        train_df = pd.DataFrame(self.X_)
        self.metadata_ = SingleTableMetadata()
        self.metadata_.detect_from_dataframe(train_df)
        self.model_ = CopulaGANSynthesizer(self.metadata_, **self.model_kwargs_)
        self.model_.fit(train_df)
        return self

    def sample(self, n_samples=1):
        sampled = self.model_.sample(num_rows=n_samples)
        return sampled.to_numpy()
