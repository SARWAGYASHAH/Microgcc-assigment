"""
Model Selector
Trains all models for a given state, evaluates on validation set,
and selects the best model based on the primary metric.
"""
import pandas as pd
import numpy as np
import logging
import json
import os
import traceback

from src.models.arima_model import SARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.utils import evaluate_forecast
import config

logger = logging.getLogger(__name__)


class ModelSelector:
    """Orchestrates training, evaluation, and selection of the best model per state."""

    def __init__(self):
        self.results = {}       # {state: {model_name: {metrics, predictions}}}
        self.best_models = {}   # {state: model_name}
        self.trained_models = {}  # {state: {model_name: model_instance}}

    def _get_model_instances(self) -> list:
        """Create fresh instances of all models."""
        return [
            SARIMAModel(),
            ProphetModel(),
            XGBoostModel(),
            LSTMModel(),
        ]

    def train_and_evaluate_state(
        self,
        state: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> dict:
        """
        Train all models for a single state, evaluate on validation set,
        and return comparison results.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  Processing State: {state}")
        logger.info(f"{'='*60}")

        state_results = {}
        state_models = {}
        y_true = val_df[config.TARGET_COL].values

        for model in self._get_model_instances():
            try:
                logger.info(f"\n--- Training {model.name} for {state} ---")

                # Train
                model.fit(train_df, val_df)

                # Predict validation
                y_pred = model.predict_validation(val_df)

                # Ensure predictions match validation length
                if len(y_pred) != len(y_true):
                    logger.warning(
                        f"[{model.name}] Prediction length mismatch: "
                        f"got {len(y_pred)}, expected {len(y_true)}. Trimming/padding."
                    )
                    if len(y_pred) > len(y_true):
                        y_pred = y_pred[:len(y_true)]
                    else:
                        # Pad with last prediction
                        y_pred = np.pad(y_pred, (0, len(y_true) - len(y_pred)), mode="edge")

                # Evaluate
                metrics = evaluate_forecast(y_true, y_pred)
                metrics["model"] = model.name

                state_results[model.name] = {
                    "metrics": metrics,
                    "val_predictions": y_pred.tolist(),
                }
                state_models[model.name] = model

                logger.info(
                    f"[{model.name}] Validation — MAE: {metrics['mae']:.2f}, "
                    f"RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%"
                )

            except Exception as e:
                logger.error(f"[{model.name}] FAILED for {state}: {str(e)}")
                logger.debug(traceback.format_exc())
                state_results[model.name] = {
                    "metrics": {
                        "mae": float("inf"),
                        "rmse": float("inf"),
                        "mape": float("inf"),
                        "smape": float("inf"),
                        "model": model.name,
                        "error": str(e),
                    },
                    "val_predictions": [],
                }

        # Select best model
        best_name = min(
            state_results,
            key=lambda m: state_results[m]["metrics"].get(config.PRIMARY_METRIC, float("inf")),
        )
        best_metric = state_results[best_name]["metrics"][config.PRIMARY_METRIC]
        logger.info(f"\n>>> Best model for {state}: {best_name} ({config.PRIMARY_METRIC}: {best_metric:.2f}%)")

        self.results[state] = state_results
        self.best_models[state] = best_name
        self.trained_models[state] = state_models

        return state_results

    def get_best_model(self, state: str):
        """Return the best trained model instance for a state."""
        if state not in self.best_models:
            raise ValueError(f"No models trained for state: {state}")
        best_name = self.best_models[state]
        return self.trained_models[state][best_name]

    def get_comparison_table(self) -> pd.DataFrame:
        """Create a comparison DataFrame across all states and models."""
        rows = []
        for state, model_results in self.results.items():
            for model_name, result in model_results.items():
                row = {"state": state, "model": model_name}
                row.update(result["metrics"])
                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def save_results(self, path: str = None):
        """Save comparison results and best model selections to JSON."""
        path = path or os.path.join(config.RESULTS_DIR, "model_comparison.json")

        output = {
            "best_models": self.best_models,
            "primary_metric": config.PRIMARY_METRIC,
            "detailed_results": {},
        }

        for state, model_results in self.results.items():
            output["detailed_results"][state] = {}
            for model_name, result in model_results.items():
                output["detailed_results"][state][model_name] = {
                    "metrics": {k: v for k, v in result["metrics"].items() if k != "model"},
                }

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved to {path}")
        return path
