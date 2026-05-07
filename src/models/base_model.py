"""
Abstract Base Model
All forecasting models inherit from this class.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract interface for all forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.model = None
        logger.info(f"[{self.name}] Initialized")

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train the model on the training data."""
        pass

    @abstractmethod
    def predict(self, horizon: int, last_known_data: pd.DataFrame = None) -> np.ndarray:
        """Generate predictions for `horizon` future steps."""
        pass

    @abstractmethod
    def predict_validation(self, val_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the validation set (for evaluation)."""
        pass

    def __repr__(self):
        return f"{self.name}(fitted={self.is_fitted})"
