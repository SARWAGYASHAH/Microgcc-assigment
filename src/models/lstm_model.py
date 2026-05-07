"""
LSTM Model (Deep Learning)
Uses Keras/TensorFlow for sequence-based time series forecasting.
"""
import pandas as pd
import numpy as np
import logging
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

from src.models.base_model import BaseForecaster
import config

logger = logging.getLogger(__name__)


class LSTMModel(BaseForecaster):
    """LSTM-based sequence forecaster using Keras."""

    def __init__(self):
        super().__init__(name="LSTM")
        self._scaler = None
        self._train_series = None
        self._seq_len = config.LSTM_SEQUENCE_LENGTH

    def _scale_data(self, data: np.ndarray) -> np.ndarray:
        """Min-Max scale the data to [0, 1]."""
        from sklearn.preprocessing import MinMaxScaler

        if self._scaler is None:
            self._scaler = MinMaxScaler(feature_range=(0, 1))
            return self._scaler.fit_transform(data.reshape(-1, 1)).flatten()
        return self._scaler.transform(data.reshape(-1, 1)).flatten()

    def _inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """Inverse the scaling."""
        return self._scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    def _create_sequences(self, data: np.ndarray, seq_len: int):
        """Create input sequences and corresponding targets for LSTM."""
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Build and train LSTM model."""
        import tensorflow as tf
        from tensorflow import keras

        # Suppress TF logging
        tf.get_logger().setLevel("ERROR")

        y_train = train_df[config.TARGET_COL].values.astype(float)
        self._train_series = y_train.copy()

        # Scale data
        scaled = self._scale_data(y_train)

        # Create sequences
        X_train, y_train_seq = self._create_sequences(scaled, self._seq_len)

        if len(X_train) < 10:
            logger.warning(f"[{self.name}] Very few training sequences ({len(X_train)}). Results may be poor.")

        # Reshape for LSTM: (samples, timesteps, features)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Prepare validation data if available
        callbacks = []
        validation_data = None

        if val_df is not None and len(val_df) > 0:
            full_series = np.concatenate([self._train_series, val_df[config.TARGET_COL].values.astype(float)])
            scaled_full = self._scaler.transform(full_series.reshape(-1, 1)).flatten()
            X_full, y_full = self._create_sequences(scaled_full, self._seq_len)
            X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], 1)

            # Validation set = sequences that start from train period but predict into val period
            val_start = len(X_train)
            if val_start < len(X_full):
                X_val = X_full[val_start:]
                y_val = y_full[val_start:]
                validation_data = (X_val, y_val)

        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data is not None else "loss",
                patience=config.LSTM_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )
        )

        # Build LSTM model
        logger.info(f"[{self.name}] Building LSTM model (seq_len={self._seq_len}, units={config.LSTM_UNITS})")

        self.model = keras.Sequential([
            keras.layers.Input(shape=(self._seq_len, 1)),
            keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True),
            keras.layers.Dropout(config.LSTM_DROPOUT),
            keras.layers.LSTM(config.LSTM_UNITS // 2, return_sequences=False),
            keras.layers.Dropout(config.LSTM_DROPOUT),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LSTM_LEARNING_RATE),
            loss="mse",
        )

        logger.info(f"[{self.name}] Training for up to {config.LSTM_EPOCHS} epochs...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history = self.model.fit(
                X_train,
                y_train_seq,
                epochs=config.LSTM_EPOCHS,
                batch_size=config.LSTM_BATCH_SIZE,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0,
            )

        actual_epochs = len(history.history["loss"])
        final_loss = history.history["loss"][-1]
        logger.info(f"[{self.name}] Trained for {actual_epochs} epochs, final loss: {final_loss:.6f}")

        self.is_fitted = True
        return self

    def predict(self, horizon: int = None, last_known_data: pd.DataFrame = None) -> np.ndarray:
        """
        Recursive multi-step forecasting.
        Uses the last `seq_len` values as seed, predicts one step at a time.
        """
        horizon = horizon or config.FORECAST_HORIZON
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Use provided data or fall back to training data
        if last_known_data is not None:
            series = last_known_data[config.TARGET_COL].values.astype(float)
        else:
            series = self._train_series

        # Scale and get the last sequence
        scaled = self._scaler.transform(series.reshape(-1, 1)).flatten()
        current_seq = scaled[-self._seq_len:].tolist()

        predictions = []
        for _ in range(horizon):
            X_input = np.array(current_seq[-self._seq_len:]).reshape(1, self._seq_len, 1)
            pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]
            predictions.append(pred_scaled)
            current_seq.append(pred_scaled)

        # Inverse scale
        return self._inverse_scale(np.array(predictions))

    def predict_validation(self, val_df: pd.DataFrame) -> np.ndarray:
        """Predict validation set values one at a time using actual history."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        full_series = np.concatenate([self._train_series, val_df[config.TARGET_COL].values.astype(float)])
        scaled_full = self._scaler.transform(full_series.reshape(-1, 1)).flatten()

        predictions = []
        for i in range(len(self._train_series), len(full_series)):
            if i < self._seq_len:
                predictions.append(full_series[i])  # Not enough history
                continue
            X_input = scaled_full[i - self._seq_len : i].reshape(1, self._seq_len, 1)
            pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]
            predictions.append(self._scaler.inverse_transform([[pred_scaled]])[0, 0])

        return np.array(predictions)
