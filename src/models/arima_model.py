"""
SARIMA Model
Uses pmdarima's auto_arima for automatic order selection with seasonal components.
"""
import pandas as pd
import numpy as np
import logging
import warnings

from src.models.base_model import BaseForecaster
import config

logger = logging.getLogger(__name__)


class SARIMAModel(BaseForecaster):
    """SARIMA forecaster using pmdarima auto_arima."""

    def __init__(self):
        super().__init__(name="SARIMA")
        self._train_series = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Fit SARIMA model using auto_arima on the training target series."""
        import pmdarima as pm

        y = train_df[config.TARGET_COL].values
        self._train_series = y

        logger.info(f"[{self.name}] Fitting auto_arima on {len(y)} observations...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use m=13 (quarterly seasonality) for speed; m=52 is too slow
            seasonal_m = min(13, len(y) // 3)
            self.model = pm.auto_arima(
                y,
                seasonal=True,
                m=seasonal_m,
                max_p=config.SARIMA_MAX_ORDER,
                max_q=config.SARIMA_MAX_ORDER,
                max_P=1,
                max_Q=1,
                max_d=2,
                max_D=1,
                stepwise=True,
                suppress_warnings=config.SARIMA_SUPPRESS_WARNINGS,
                error_action="ignore",
                trace=False,
                n_fits=20,
            )

        self.is_fitted = True
        logger.info(f"[{self.name}] Best order: {self.model.order}, seasonal: {self.model.seasonal_order}")
        logger.info(f"[{self.name}] AIC: {self.model.aic():.2f}")
        return self

    def predict(self, horizon: int = None, last_known_data: pd.DataFrame = None) -> np.ndarray:
        """Forecast `horizon` steps ahead."""
        horizon = horizon or config.FORECAST_HORIZON
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        preds = self.model.predict(n_periods=horizon)
        return np.array(preds)

    def predict_validation(self, val_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the validation period."""
        horizon = len(val_df)
        return self.predict(horizon=horizon)
