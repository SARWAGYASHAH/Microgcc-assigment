"""
XGBoost Model
Gradient boosted trees with engineered lag, rolling, and calendar features.
"""
import pandas as pd
import numpy as np
import logging
import joblib
import os

from src.models.base_model import BaseForecaster
from src.feature_engineering import engineer_features, get_feature_columns, prepare_ml_data
import config

logger = logging.getLogger(__name__)


class XGBoostModel(BaseForecaster):
    """XGBoost regressor with time-series features."""

    def __init__(self):
        super().__init__(name="XGBoost")
        self._feature_cols = None
        self._last_train_data = None  # Keep for recursive forecasting

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Fit XGBoost on engineered features."""
        from xgboost import XGBRegressor

        # Engineer features on train set
        train_feat = engineer_features(train_df)
        self._feature_cols = get_feature_columns(train_feat)

        X_train, y_train = prepare_ml_data(train_feat, drop_na=True)

        # Prepare validation set for early stopping
        eval_set = None
        if val_df is not None and len(val_df) > 0:
            val_feat = engineer_features(pd.concat([train_df, val_df], ignore_index=True))
            # Only take the validation portion (after dropping NaN from lags)
            val_start_idx = len(train_feat)
            val_feat_only = val_feat.iloc[val_start_idx:].dropna(subset=self._feature_cols)
            if len(val_feat_only) > 0:
                X_val = val_feat_only[self._feature_cols]
                y_val = val_feat_only[config.TARGET_COL]
                eval_set = [(X_val, y_val)]

        logger.info(f"[{self.name}] Training on {len(X_train)} samples, {len(self._feature_cols)} features")

        xgb_params = dict(config.XGBOOST_PARAMS)
        if eval_set:
            xgb_params["early_stopping_rounds"] = config.XGBOOST_EARLY_STOPPING

        self.model = XGBRegressor(**xgb_params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        self.is_fitted = True
        self._last_train_data = train_df.copy()

        if hasattr(self.model, "best_iteration"):
            logger.info(f"[{self.name}] Best iteration: {self.model.best_iteration}")

        # Log feature importance (top 10)
        importances = self.model.feature_importances_
        feat_imp = sorted(zip(self._feature_cols, importances), key=lambda x: x[1], reverse=True)
        logger.info(f"[{self.name}] Top 5 features: {feat_imp[:5]}")

        return self

    def predict(self, horizon: int = None, last_known_data: pd.DataFrame = None) -> np.ndarray:
        """
        Recursive multi-step forecasting.
        Uses the last known data to generate features, then predicts step by step.
        """
        horizon = horizon or config.FORECAST_HORIZON
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        data = last_known_data if last_known_data is not None else self._last_train_data
        data = data.copy()

        predictions = []
        for step in range(horizon):
            # Generate next date
            last_date = data[config.DATE_COL].max()
            next_date = last_date + pd.Timedelta(weeks=1)

            # Create a new row with the next date (target unknown)
            new_row = pd.DataFrame({
                config.DATE_COL: [next_date],
                config.TARGET_COL: [np.nan],
                config.STATE_COL: [data[config.STATE_COL].iloc[0]],
                config.CATEGORY_COL: [data[config.CATEGORY_COL].iloc[0]],
            })
            data = pd.concat([data, new_row], ignore_index=True)

            # Re-engineer features on entire history + new row
            data_feat = engineer_features(data)
            last_row = data_feat.iloc[[-1]]

            # Fill any NaN features with 0 for prediction
            X_pred = last_row[self._feature_cols].fillna(0)
            pred = self.model.predict(X_pred)[0]

            # Store prediction and update the target for next step's lag features
            predictions.append(pred)
            data.iloc[-1, data.columns.get_loc(config.TARGET_COL)] = pred

        return np.array(predictions)

    def predict_validation(self, val_df: pd.DataFrame) -> np.ndarray:
        """Predict on validation set using actual features (not recursive)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Combine train + val for proper lag computation
        combined = pd.concat([self._last_train_data, val_df], ignore_index=True)
        combined_feat = engineer_features(combined)

        # Extract only validation rows
        val_start = len(self._last_train_data)
        val_feat = combined_feat.iloc[val_start:]

        X_val = val_feat[self._feature_cols].fillna(0)
        return self.model.predict(X_val)

    def save(self, path: str):
        """Save the model to disk."""
        joblib.dump({"model": self.model, "feature_cols": self._feature_cols}, path)
        logger.info(f"[{self.name}] Saved to {path}")

    def load(self, path: str):
        """Load the model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self._feature_cols = data["feature_cols"]
        self.is_fitted = True
        logger.info(f"[{self.name}] Loaded from {path}")
