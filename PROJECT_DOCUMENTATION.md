# Project Documentation: End-to-End Time Series Forecasting System

## 1. Project Overview

This project forecasts the next 8 weeks of beverage sales for each US state using historical sales data from the provided Excel file, `Forecasting Case- Study.xlsx`.

The system is designed like a backend forecasting service rather than a notebook-only experiment. It includes a training pipeline, model comparison logic, generated artifacts, a Flask REST API, and a dashboard UI.

## 2. Problem Statement

The goal is to build a production-ready forecasting system that:

- Trains multiple forecasting algorithms.
- Compares model performance by state.
- Selects the best model automatically.
- Forecasts the next 8 weeks of sales for each state.
- Exposes predictions through REST API endpoints.
- Handles missing dates, missing values, trend, and seasonality.

## 3. Dataset

The input dataset is an Excel file:

```text
Forecasting Case- Study.xlsx
```

The project expects these columns, configured in `config.py`:

| Config Value | Column |
|--------------|--------|
| `DATE_COL` | `Date` |
| `TARGET_COL` | `Total` |
| `STATE_COL` | `State` |
| `CATEGORY_COL` | `Category` |

The target variable is `Total`, which represents beverage sales.

## 4. System Flow

The pipeline flow is:

```text
Raw Excel data
    -> Data loading
    -> Date parsing
    -> Missing value handling
    -> Weekly resampling per state
    -> Time-series train/validation split
    -> Model training
    -> Validation prediction
    -> Metric calculation
    -> Best model selection
    -> 8-week forecast generation
    -> Artifact saving
    -> API serving
    -> Dashboard visualization
```

## 5. Data Loading and Cleaning

Implemented in:

```text
src/data_loader.py
```

Main functions:

| Function | Responsibility |
|----------|----------------|
| `load_raw_data()` | Reads the Excel dataset |
| `clean_data()` | Parses dates, sorts rows, fills missing values, resamples weekly |
| `get_state_data()` | Extracts one state's time series |
| `train_val_split()` | Splits data using the last 8 weeks as validation |
| `load_and_prepare()` | Runs load and clean steps together |

Cleaning decisions:

- Dates are parsed using `pd.to_datetime(..., format="mixed", dayfirst=True)`.
- Data is sorted by state and date.
- Missing target values are forward-filled and backward-filled within each state.
- Each state is resampled to weekly Sunday-ending frequency.
- Missing weekly periods are linearly interpolated.
- Remaining leading missing values are back-filled.

## 6. Feature Engineering

Implemented in:

```text
src/feature_engineering.py
```

The project creates the required assignment features and extra features useful for ML models.

### Lag Features

Configured in `config.py`:

```python
LAG_WEEKS = [1, 2, 3, 4, 7, 13, 30]
```

This includes the required:

- `t-1`
- `t-7`
- `t-30`

Because the data is resampled weekly, these lags represent weekly periods.

### Rolling Features

Configured in `config.py`:

```python
ROLLING_WINDOWS = [4, 8, 13]
```

For each window, the pipeline creates:

- `rolling_mean_<window>`
- `rolling_std_<window>`

Rolling features are shifted by one period to avoid leakage.

### Calendar Features

Calendar features include:

- Week of year
- Month
- Quarter
- Year
- Day of week
- Day of year
- Month start flag
- Month end flag

### Holiday Features

The project uses the `holidays` package to create US holiday flags.

Features:

- `is_holiday`
- `holiday_flag`

### Trend Features

Trend features include:

- `time_index`
- `time_index_norm`

## 7. Train/Validation Strategy

Implemented in:

```text
src/data_loader.py
```

The project uses a time-series split:

- Training data: all observations except the final 8 weeks.
- Validation data: final 8 weeks.

This prevents data leakage because the model never trains on future observations.

The validation window is configured in `config.py`:

```python
VALIDATION_WEEKS = 8
```

## 8. Models

All models inherit from:

```text
src/models/base_model.py
```

Each model implements:

- `fit()`
- `predict()`
- `predict_validation()`

### SARIMA

File:

```text
src/models/arima_model.py
```

Implementation:

- Uses `pmdarima.auto_arima`.
- Uses seasonal ARIMA components.
- Automatically searches model orders.

### Prophet

File:

```text
src/models/prophet_model.py
```

Implementation:

- Uses Facebook Prophet.
- Enables yearly seasonality.
- Adds US holidays.
- Handles trend and changepoints.

### XGBoost

File:

```text
src/models/xgboost_model.py
```

Implementation:

- Uses `xgboost.XGBRegressor`.
- Trains on engineered lag, rolling, calendar, holiday, and trend features.
- Uses recursive multi-step forecasting for future periods.

### LSTM

File:

```text
src/models/lstm_model.py
```

Implementation:

- Uses TensorFlow/Keras.
- Applies MinMax scaling.
- Builds sequence windows from historical sales.
- Uses a 2-layer LSTM architecture with dropout.
- Forecasts recursively for the requested horizon.

## 9. Model Selection

Implemented in:

```text
src/model_selector.py
```

The `ModelSelector` class:

- Creates fresh instances of all models.
- Trains all models for a state.
- Generates validation predictions.
- Computes metrics.
- Selects the best model using the configured primary metric.

The primary metric is configured in `config.py`:

```python
PRIMARY_METRIC = "mape"
```

Lower MAPE is better.

## 10. Evaluation Metrics

Implemented in:

```text
src/utils.py
```

Metrics:

| Metric | Meaning |
|--------|---------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| sMAPE | Symmetric Mean Absolute Percentage Error |

MAPE is used for model selection.

## 11. Training Pipeline

Implemented in:

```text
main.py
```

Main function:

```python
run_pipeline(states=None)
```

Pipeline steps:

1. Set up logging.
2. Load and clean the dataset.
3. Get all states or selected states.
4. Split each state's time series into train and validation data.
5. Train SARIMA, Prophet, XGBoost, and LSTM.
6. Evaluate each model on validation data.
7. Select the best model for that state.
8. Retrain the best model on all available state data.
9. Generate the next 8 weeks of forecasts.
10. Save forecasts, comparison metrics, trained models, and prepared data.

Command examples:

```bash
python main.py --test
python main.py --states California Texas "New York"
python main.py
```

## 12. Generated Artifacts

Artifacts are saved under:

```text
artifacts/
```

| File | Description |
|------|-------------|
| `artifacts/results/forecasts.json` | Forecast dates, forecast values, and best model by state |
| `artifacts/results/model_comparison.json` | Best model and detailed metrics by state |
| `artifacts/results/comparison_table.csv` | Flat comparison table for all states and models |
| `artifacts/models/trained_models.pkl` | Serialized trained best models |
| `artifacts/prepared_data.pkl` | Cleaned weekly data for API historical endpoint |

Large binary artifacts are ignored by Git. Run the pipeline to regenerate them.

## 13. REST API

Implemented in:

```text
api/app.py
```

The API loads saved artifacts at startup.

Start the API:

```bash
python -m api.app
```

Default URL:

```text
http://localhost:5000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API metadata and endpoint list |
| GET | `/health` | Health check and artifact availability |
| GET | `/states` | Available states |
| GET | `/forecast/<state>` | Saved 8-week forecast for a state |
| GET | `/forecast/all` | Saved forecasts for all states |
| GET | `/models/comparison` | Detailed model comparison metrics |
| GET | `/models/best` | Best model per state |
| POST | `/predict` | On-demand forecast using trained model objects |
| GET | `/historical/<state>` | Historical weekly sales for a state |
| GET | `/dashboard` | Browser dashboard |

### Example Requests

Forecast for California:

```bash
curl http://localhost:5000/forecast/California
```

Best models:

```bash
curl http://localhost:5000/models/best
```

On-demand prediction:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"state\": \"Texas\", \"horizon\": 12}"
```

## 14. Dashboard UI

Implemented in:

```text
api/static/
```

Files:

| File | Purpose |
|------|---------|
| `index.html` | Dashboard markup |
| `styles.css` | Dashboard styling |
| `app.js` | API calls and chart rendering |

Open after starting the API:

```text
http://localhost:5000/dashboard
```

Dashboard features:

- State selector
- API health status
- Historical and forecast chart
- 8-week forecast table
- Best model and MAPE summary
- Model comparison table
- MAPE chart by model
- Remaining work checklist

## 15. Configuration

Central configuration lives in:

```text
config.py
```

Important settings:

| Setting | Purpose |
|---------|---------|
| `DATA_PATH` | Excel input path |
| `FORECAST_HORIZON` | Number of weeks to forecast |
| `VALIDATION_WEEKS` | Validation window size |
| `LAG_WEEKS` | Lag feature periods |
| `ROLLING_WINDOWS` | Rolling feature windows |
| `PRIMARY_METRIC` | Metric used for best model selection |
| `API_HOST` | Flask host |
| `API_PORT` | Flask port |

## 16. How to Reproduce Results

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Run quick test.

```bash
python main.py --test
```

3. Run full training.

```bash
python main.py
```

4. Start API.

```bash
python -m api.app
```

5. Open dashboard.

```text
http://localhost:5000/dashboard
```

## 17. Known Limitations

- There are no automated tests yet.
- The Flask app currently uses the development server.
- Large trained model artifacts are not committed to Git.
- Full training can be slow because it trains four model families for every state.
- The LSTM can produce slightly different results across runs unless all random seeds and TensorFlow settings are fully controlled.

## 18. Recommended Next Steps

- Add unit tests for data cleaning and feature engineering.
- Add API integration tests.
- Add Dockerfile and production startup command.
- Add a retraining schedule for new data.
- Add model monitoring for forecast drift.
- Add authentication if the API is exposed outside a private environment.
