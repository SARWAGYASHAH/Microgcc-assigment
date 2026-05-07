# End-to-End Time Series Forecasting System

This project is a production-style forecasting service for predicting the next 8 weeks of beverage sales for each US state from the provided Excel dataset.

It trains multiple forecasting models, compares them using validation metrics, selects the best model per state, saves forecast artifacts, exposes predictions through a Flask REST API, and includes a browser dashboard for reviewing results.

For the full project explanation, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## Key Features

- Loads and cleans the provided Excel dataset.
- Handles missing dates by resampling each state to a regular weekly frequency.
- Handles missing sales values through interpolation and forward/backward filling.
- Creates lag, rolling, calendar, holiday, and trend features.
- Uses a time-series validation split with the last 8 weeks as validation data.
- Trains and compares SARIMA, Prophet, XGBoost, and LSTM models.
- Selects the best model per state using MAPE.
- Generates 8-week forecasts for every state.
- Serves forecasts, model metrics, historical data, and predictions through a REST API.
- Provides a dashboard UI at `/dashboard`.

## Architecture

```text
Excel Dataset
    |
    v
Data Loader
    - parse dates
    - clean missing values
    - resample weekly per state
    |
    v
Feature Engineering
    - lags
    - rolling mean/std
    - calendar fields
    - holiday flags
    - trend features
    |
    v
Model Training per State
    - SARIMA
    - Prophet
    - XGBoost
    - LSTM
    |
    v
Model Selector
    - evaluate on validation data
    - select lowest MAPE
    |
    v
Artifacts + Flask API + Dashboard
```

## Models Implemented

| Model | Type | Purpose |
|-------|------|---------|
| SARIMA | Statistical | Captures autoregressive and seasonal time-series behavior |
| Prophet | Statistical | Captures trend, yearly seasonality, changepoints, and holidays |
| XGBoost | Machine Learning | Uses engineered lag, rolling, calendar, holiday, and trend features |
| LSTM | Deep Learning | Learns sequential patterns from recent sales history |

## Feature Engineering

The feature pipeline is implemented in `src/feature_engineering.py`.

| Feature Group | Details |
|---------------|---------|
| Lag features | `t-1`, `t-2`, `t-3`, `t-4`, `t-7`, `t-13`, `t-30` weekly periods |
| Rolling statistics | Rolling mean and standard deviation over 4, 8, and 13 weeks |
| Calendar features | Week of year, month, quarter, year, day of week, day of year |
| Cyclical features | Sine/cosine encodings for month and week of year |
| Holiday features | US holiday flag using the `holidays` library |
| Trend features | Time index and normalized time index |

## Assignment Coverage

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Forecast next 8 weeks for each state | Done | `main.py`, `/forecast/<state>` |
| Handle missing dates and missing values | Done | `src/data_loader.py` |
| Handle seasonality and trend | Done | SARIMA, Prophet, calendar/holiday/trend features |
| Automatically select the best model | Done | `src/model_selector.py` |
| Implement ARIMA/SARIMA | Done | `src/models/arima_model.py` |
| Implement Facebook Prophet | Done | `src/models/prophet_model.py` |
| Implement XGBoost with lag features | Done | `src/models/xgboost_model.py` |
| Implement LSTM | Done | `src/models/lstm_model.py` |
| Create lag features t-1, t-7, t-30 | Done | `config.py`, `src/feature_engineering.py` |
| Create rolling mean/std | Done | `src/feature_engineering.py` |
| Create day of week, month, holiday flag | Done | `src/feature_engineering.py` |
| Use time-series validation split | Done | `src/data_loader.py` |
| Serve predictions with REST API | Done | `api/app.py` |
| Real backend-service structure | Done | Modular source folders, config, artifacts, API layer |

## Project Structure

```text
assigmnet/
|-- api/
|   |-- __init__.py
|   |-- app.py
|   `-- static/
|       |-- app.js
|       |-- index.html
|       `-- styles.css
|-- artifacts/
|   `-- results/
|       |-- comparison_table.csv
|       |-- forecasts.json
|       `-- model_comparison.json
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- feature_engineering.py
|   |-- model_selector.py
|   |-- utils.py
|   `-- models/
|       |-- __init__.py
|       |-- arima_model.py
|       |-- base_model.py
|       |-- lstm_model.py
|       |-- prophet_model.py
|       `-- xgboost_model.py
|-- config.py
|-- main.py
|-- requirements.txt
|-- Forecasting Case- Study.xlsx
|-- PROJECT_DOCUMENTATION.md
`-- README.md
```

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Training Pipeline

Quick test on three states:

```bash
python main.py --test
```

Train specific states:

```bash
python main.py --states California Texas "New York"
```

Train all states:

```bash
python main.py
```

The pipeline saves outputs under `artifacts/`.

## Start the API

```bash
python -m api.app
```

Open the dashboard:

```text
http://localhost:5000/dashboard
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API documentation and service metadata |
| GET | `/health` | Health check |
| GET | `/states` | List available states |
| GET | `/forecast/<state>` | Return saved 8-week forecast for one state |
| GET | `/forecast/all` | Return saved forecasts for all states |
| GET | `/models/comparison` | Return detailed model comparison metrics |
| GET | `/models/best` | Return best model selected per state |
| POST | `/predict` | Generate on-demand prediction using trained model artifacts |
| GET | `/historical/<state>` | Return historical sales data for one state |
| GET | `/dashboard` | Browser dashboard UI |

Example:

```bash
curl http://localhost:5000/forecast/California
```

On-demand prediction:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"state\": \"Texas\", \"horizon\": 12}"
```

## Generated Artifacts

| Artifact | Purpose |
|----------|---------|
| `artifacts/results/forecasts.json` | Saved 8-week forecasts by state |
| `artifacts/results/model_comparison.json` | Best models and metrics |
| `artifacts/results/comparison_table.csv` | Flat table of validation metrics |
| `artifacts/models/trained_models.pkl` | Trained model objects for `/predict` |
| `artifacts/prepared_data.pkl` | Prepared historical data for API historical endpoint |

Large binary artifacts are ignored by Git. If you clone the project fresh, run `python main.py` to regenerate `trained_models.pkl` and `prepared_data.pkl`.

## Evaluation

Models are evaluated on the final 8 weeks of each state's time series. This avoids leakage because validation data is always later than training data.

Metrics:

- MAPE, the primary selection metric
- MAE
- RMSE
- sMAPE

## Dashboard

The dashboard is served by Flask from `api/static/` and displays:

- State selector
- API health status
- Historical sales trend
- 8-week forecast line
- Forecast value list
- Best model and MAPE
- Model comparison table and chart
- Remaining work checklist

## Remaining Work

- Add automated tests for data loading, feature engineering, metrics, model selection, and API responses.
- Add a Dockerfile and production WSGI server configuration.
- Add model monitoring and scheduled retraining notes.
- Re-run the full pipeline after feature changes so saved model artifacts match the latest feature set.
