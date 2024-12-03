import numpy as np
from .base_generator import BaseGenerator

class GenDummy(BaseGenerator):
    """A dummy generator that returns the original dataset as is."""

    def __init__(self):
        """Initialize the generator."""
        super().__init__()
        self.name_ = "dummy"

    def fit(self, X: np.ndarray, y: np.ndarray = None, metamodel=None) -> "GenDummy":
        """Fit the generator to the input data.
        Args:
            X (np.ndarray): Input data to fit the generator on.
            y (np.ndarray): Target labels (not used).
            metamodel: Additional metamodel information (not used).
        Returns:
            GenDummy: The fitted generator instance.
        """
        self.X_ = X.copy()
        return self

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate samples by returning a copy of the fitted data."""
        if self.X_ is None:
            raise ValueError("Generator must be fitted before sampling.")
        return self.X_.copy()

    def my_name(self) -> str:
        """Return the name of the generator."""
        return self.name_

