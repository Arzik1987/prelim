from abc import ABC, abstractmethod
import numpy as np

class BaseGenerator(ABC):
    """Abstract base class for data generators."""
    
    def __init__(self):
        self.X_ = None
        self.name_ = "base"

    @abstractmethod
    def fit(self, X: np.ndarray):
        """Fit the generator to the input data."""
        pass

    @abstractmethod
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate samples."""
        pass

    def my_name(self) -> str:
        """Return the name of the generator."""
        return self.name_
