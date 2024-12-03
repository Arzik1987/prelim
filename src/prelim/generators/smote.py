import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from .base_generator import BaseGenerator
from .rand import Gen_randu


class GenSmote(BaseGenerator):
    """Data generator using SMOTE for oversampling."""

    def __init__(self):
        super().__init__()
        self.name_ = "smote"

    def fit(self, X: np.ndarray, y: np.ndarray = None, metamodel=None) -> "GenSmote":
        """Fit the generator to the input data.
        Args:
            X (np.ndarray): Input feature matrix.
            y (np.ndarray): Target labels (not used).
            metamodel: Additional metamodel information (not used).
        Returns:
            GenSmote: Instance of the fitted generator.
        """
        self.X_ = X.copy()
        return self

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate samples using SMOTE."""
        if self.X_ is None:
            raise ValueError("Generator must be fitted before sampling.")

        # Determine sampling strategy
        sampling_strategy = 'not majority'
        if self.X_.shape[0] > n_samples:
            warnings.warn(
                "The required sample size is smaller than the number of observations in training."
            )
            sampling_strategy = 'all'

        # Generate initial synthetic samples
        y = np.concatenate((np.ones(self.X_.shape[0]), np.zeros(n_samples)))
        X = np.concatenate((self.X_, Gen_randu().fit(self.X_).sample(n_samples=n_samples)))

        # Use SMOTE to generate balanced samples
        k_neighbors = min(5, n_samples, self.X_.shape[0])
        X_new, y_new = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=2020
        ).fit_resample(X, y)

        # Return only the generated samples
        return X_new[y_new == 1][:n_samples]

    def my_name(self) -> str:
        return self.name_

