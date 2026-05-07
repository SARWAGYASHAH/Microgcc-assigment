# Time Series Forecasting System

A production-ready, end-to-end forecasting system for beverage sales across 43 US states.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────┐    ┌──────────────┐
│  Excel Data │───▶│  Data Loader     │───▶│  Feature Eng  │───▶│  4 Models    │
│  (8K rows)  │    │  (Clean/Resample)│    │  (Lags/Roll)  │    │  per State   │
└─────────────┘    └──────────────────┘    └───────────────┘    └──────┬───────┘
                                                                       │
                   ┌──────────────────┐    ┌───────────────┐          │
                   │   REST API       │◀───│ Model Selector│◀─────────┘
                   │   (Flask)        │    │ (Best MAPE)   │
                   └──────────────────┘    └───────────────┘
```

## Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| **SARIMA** | Statistical | Auto-ARIMA with seasonal components (pmdarima) |
| **Prophet** | Statistical | Facebook Prophet with US holidays & yearly seasonality |
| **XGBoost** | ML | Gradient boosting with engineered lag/rolling/calendar features |
| **LSTM** | Deep Learning | 2-layer LSTM with Keras/TensorFlow, sequence-based |

## Feature Engineering

- **Lag features**: t-1, t-2, t-3, t-4, t-7, t-13, t-30 (weekly periods after resampling)
- **Rolling statistics**: Mean & std over 4, 8, 13 week windows
- **Calendar**: Week of year, month, quarter, day of week, cyclical encodings
- **Holiday flags**: US federal holidays (via `holidays` library)
- **Trend**: Time index, normalized time

## Assignment Coverage

| Requirement | Status | Where |
|-------------|--------|-------|
| Forecast next 8 weeks for each state | Done | `main.py`, `artifacts/results/forecasts.json`, `/forecast/<state>` |
| Handle missing dates / values | Done | `src/data_loader.py` resamples weekly and fills/interpolates missing values |
| Handle seasonality and trend | Done | SARIMA/Prophet plus calendar, cyclical, holiday, and trend features |
| Automatically select best model | Done | `src/model_selector.py` selects lowest MAPE per state |
| Implement SARIMA | Done | `src/models/arima_model.py` |
| Implement Facebook Prophet | Done | `src/models/prophet_model.py` |
| Implement XGBoost with lag features | Done | `src/models/xgboost_model.py`, `src/feature_engineering.py` |
| Implement LSTM | Done | `src/models/lstm_model.py` |
| Lag features t-1, t-7, t-30 | Done | `config.py`, `src/feature_engineering.py` |
| Rolling mean / std | Done | `src/feature_engineering.py` |
| Day of week, month, holiday flag | Done | `src/feature_engineering.py` |
| Time-series validation split | Done | `src/data_loader.py` uses the last 8 weeks as validation |
| REST API backend service | Done | `api/app.py` |
| Browser UI | Done | `/dashboard` |

## Remaining Work

- Add automated tests for data cleaning, feature engineering, model selection, and API response schemas.
- Add deployment packaging such as Dockerfile, environment file, and production WSGI startup command.
- Add model monitoring and retraining notes for future data refreshes.
- Re-run `python main.py` after changing lag settings if you want saved model artifacts trained with the new `lag_30` feature.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Quick Test — 3 states)
```bash
python main.py --test
```

### 3. Train All States
```bash
python main.py
```

### 4. Start the API
```bash
python -m api.app
```

### 5. Open the UI
After the API starts, open:

```text
http://localhost:5000/dashboard
```

The dashboard lets you choose a state, review historical sales, inspect the 8-week forecast, and compare model validation metrics.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API documentation |
| GET | `/health` | Health check |
| GET | `/states` | List all states |
| GET | `/forecast/<state>` | 8-week forecast for a state |
| GET | `/forecast/all` | All state forecasts |
| GET | `/models/comparison` | Model comparison metrics |
| GET | `/models/best` | Best model per state |
| POST | `/predict` | On-demand predictions |
| GET | `/historical/<state>` | Historical sales data |
| GET | `/dashboard` | Browser dashboard UI |

### Example API Calls

```bash
# Get forecast for California
curl http://localhost:5000/forecast/California

# On-demand prediction (custom horizon)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"state": "Texas", "horizon": 12}'

# Model comparison
curl http://localhost:5000/models/comparison
```

## Project Structure

```
assigmnet/
├── config.py                  # Central configuration & hyperparameters
├── main.py                    # Pipeline orchestrator (train → compare → forecast)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── Forecasting Case- Study.xlsx  # Raw dataset
├── api/
│   ├── __init__.py
│   └── app.py                 # Flask REST API
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading, cleaning, resampling
│   ├── feature_engineering.py # Feature creation pipeline
│   ├── utils.py               # Metrics & logging
│   ├── model_selector.py      # Model comparison & selection
│   └── models/
│       ├── __init__.py
│       ├── base_model.py      # Abstract base class
│       ├── arima_model.py     # SARIMA implementation
│       ├── prophet_model.py   # Prophet implementation
│       ├── xgboost_model.py   # XGBoost implementation
│       └── lstm_model.py      # LSTM implementation
└── artifacts/                 # Generated outputs
    ├── models/                # Saved model objects (.pkl)
    └── results/               # Forecasts, comparisons (.json, .csv)
```

## Evaluation

Models are compared using:
- **MAPE** (primary — used for model selection)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **sMAPE** (Symmetric MAPE)

Train/validation split uses the last 8 weeks as validation (time-series aware, no data leakage).
