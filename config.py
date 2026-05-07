"""
Central configuration for the Forecasting System.
All hyperparameters, paths, and constants are defined here.
"""
import os

# ─── Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Forecasting Case- Study.xlsx")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")

# Create directories
for d in [ARTIFACTS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Data ─────────────────────────────────────────────────
FREQ = "W-SUN"  # Weekly frequency, ending Sunday
DATE_COL = "Date"
TARGET_COL = "Total"
STATE_COL = "State"
CATEGORY_COL = "Category"

# ─── Forecasting ──────────────────────────────────────────
FORECAST_HORIZON = 8  # 8 weeks ahead
VALIDATION_WEEKS = 8  # Last 8 weeks of data used for validation

# ─── Feature Engineering ──────────────────────────────────
# Required assignment lags are included: t-1, t-7, and t-30.
# Because the source is resampled to weekly frequency, these are weekly periods.
LAG_WEEKS = [1, 2, 3, 4, 7, 13, 30]
ROLLING_WINDOWS = [4, 8, 13]     # 1-month, 2-month, quarter rolling windows
COUNTRY = "US"                    # For holiday detection

# ─── SARIMA ───────────────────────────────────────────────
SARIMA_SEASONAL_PERIOD = 52       # Weekly seasonality = ~52 weeks/year
SARIMA_MAX_ORDER = 2
SARIMA_SUPPRESS_WARNINGS = True

# ─── Prophet ──────────────────────────────────────────────
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = False  # Data is already weekly aggregated
PROPHET_DAILY_SEASONALITY = False
PROPHET_CHANGEPOINT_PRIOR = 0.05

# ─── XGBoost ──────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
XGBOOST_EARLY_STOPPING = 50

# ─── LSTM ─────────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH = 12    # Use last 12 weeks as input sequence
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 16
LSTM_PATIENCE = 15           # Early stopping patience
LSTM_LEARNING_RATE = 0.001

# ─── API ──────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# ─── Evaluation Metrics ──────────────────────────────────
PRIMARY_METRIC = "mape"  # Used for model selection (lower is better)
