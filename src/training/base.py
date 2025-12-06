"""Protocol definition for Machine Learning models used in training."""

from typing import Protocol

import numpy as np
import pandas as pd


class Model(Protocol):
    """Protocol for ML models supporting fit/predict/predict_proba."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions for the given data."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class probabilities for the given data."""
        ...
