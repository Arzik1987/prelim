import numpy as np
import pandas as pd

from .base import BaseGenerator

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer
except Exception:
    SingleTableMetadata = None
    GaussianCopulaSynthesizer = None


class Gen_gaussiancopula(BaseGenerator):
    def __init__(self, model_kwargs: dict | None = None, seed=2020):
        super().__init__("gaussiancopula", seed=seed)
        self.model_kwargs_ = {} if model_kwargs is None else dict(model_kwargs)
        self.X_ = None
        self.model_ = None
        self.metadata_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = np.asarray(X).copy()
        train_df = pd.DataFrame(self.X_)
        metadata_class = SingleTableMetadata
        model_class = GaussianCopulaSynthesizer
        if metadata_class is None or model_class is None:
            from sdv.metadata import SingleTableMetadata as metadata_class
            from sdv.single_table import GaussianCopulaSynthesizer as model_class

        self.metadata_ = metadata_class()
        self.metadata_.detect_from_dataframe(train_df)
        self.model_ = model_class(self.metadata_, **self.model_kwargs_)
        self.model_.fit(train_df)
        return self

    def sample(self, n_samples=1):
        sampled = self.model_.sample(num_rows=n_samples)
        return sampled.to_numpy()
