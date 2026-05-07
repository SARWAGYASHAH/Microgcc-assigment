"""
Facebook Prophet Model
Handles trend, seasonality, and holidays automatically.
"""
import pandas as pd
import numpy as np
import logging
import warnings

from src.models.base_model import BaseForecaster
import config

logger = logging.getLogger(__name__)


class ProphetModel(BaseForecaster):
    """Prophet forecaster with US holidays."""

    def __init__(self):
        super().__init__(name="Prophet")
        self._train_end_date = None

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Fit Prophet model."""
        from prophet import Prophet

        # Prophet expects columns 'ds' (date) and 'y' (target)
        prophet_df = pd.DataFrame({
            "ds": train_df[config.DATE_COL].values,
            "y": train_df[config.TARGET_COL].values,
        })

        self._train_end_date = prophet_df["ds"].max()

        logger.info(f"[{self.name}] Fitting on {len(prophet_df)} observations...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Prophet(
                yearly_seasonality=config.PROPHET_YEARLY_SEASONALITY,
                weekly_seasonality=config.PROPHET_WEEKLY_SEASONALITY,
                daily_seasonality=config.PROPHET_DAILY_SEASONALITY,
                changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR,
            )
            self.model.add_country_holidays(country_name=config.COUNTRY)
            self.model.fit(prophet_df)

        self.is_fitted = True
        logger.info(f"[{self.name}] Fitted successfully")
        return self

    def predict(self, horizon: int = None, last_known_data: pd.DataFrame = None) -> np.ndarray:
        """Forecast `horizon` weeks ahead from the end of training data."""
        horizon = horizon or config.FORECAST_HORIZON
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        future = self.model.make_future_dataframe(periods=horizon, freq=config.FREQ)
        forecast = self.model.predict(future)

        # Return only the future predictions
        preds = forecast.tail(horizon)["yhat"].values
        return np.array(preds)

    def predict_validation(self, val_df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the validation period dates."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        future = pd.DataFrame({"ds": val_df[config.DATE_COL].values})
        forecast = self.model.predict(future)
        return forecast["yhat"].values

    def get_future_dates(self, horizon: int = None) -> pd.DataFrame:
        """Get future date DataFrame from Prophet."""
        horizon = horizon or config.FORECAST_HORIZON
        future = self.model.make_future_dataframe(periods=horizon, freq=config.FREQ)
        return future.tail(horizon)
