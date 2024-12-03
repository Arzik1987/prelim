import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE
import warnings
from .base_generator import BaseGenerator
from .rand import Gen_randu


class GenAdasyn(BaseGenerator):
    """Data generator using ADASYN for oversampling."""

    def __init__(self):
        super().__init__()
        self.name_ = "adasyn"

    def fit(self, X: np.ndarray, y: np.ndarray = None, metamodel=None) -> "GenAdasyn":
        """Fit the generator to the input data.
        Args:
            X (np.ndarray): Input data to fit the generator on.
            y (np.ndarray): Target labels (not used).
            metamodel: Additional metamodel information (not used).
        Returns:
            GenAdasyn: The fitted generator instance.
        """
        self.X_ = X.copy()
        return self

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate samples using ADASYN or fallback to SMOTE."""
        if self.X_ is None:
            raise ValueError("Generator must be fitted before sampling.")

        sampling_strategy = 'not majority'
        if self.X_.shape[0] > n_samples:
            warnings.warn(
                "The required sample size is smaller than the number of observations in training."
            )
            sampling_strategy = 'all'

        # Generate initial synthetic samples
        y = np.concatenate((np.ones(self.X_.shape[0]), np.zeros(n_samples)))
        X = np.concatenate((self.X_, Gen_randu().fit(self.X_).sample(n_samples=n_samples)))

        X_new = None
        k_neighbors = min(5, n_samples, self.X_.shape[0])

        # Attempt ADASYN sampling with varying k_neighbors
        while not isinstance(X_new, np.ndarray) and k_neighbors <= n_samples and k_neighbors <= self.X_.shape[0]:
            try:
                X_new, y = ADASYN(
                    sampling_strategy=sampling_strategy,
                    n_neighbors=k_neighbors,
                    random_state=2020
                ).fit_resample(X, y)
            except (ValueError, RuntimeError):
                k_neighbors *= 2

        # Fallback to SMOTE if ADASYN fails
        if not isinstance(X_new, np.ndarray):
            k_neighbors = min(5, n_samples, self.X_.shape[0])
            X_new, y = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=2020
            ).fit_resample(X, y)
            self.name_ = "adasyn_fallback"
        else:
            self.name_ = "adasyn"

        return X_new[y == 1][:n_samples]

    def my_name(self) -> str:
        return self.name_
