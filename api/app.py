"""
Flask REST API for the Forecasting System
Serves predictions, model comparison, and health checks.
"""
import os
import sys
import json
import logging
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# ─── Flask App ────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ─── Global state (loaded on startup) ────────────────────
_forecasts = None
_comparison = None
_models_data = None
_prepared_data = None


def _load_artifacts():
    """Load pre-computed forecasts, models, and data."""
    global _forecasts, _comparison, _models_data, _prepared_data

    forecasts_path = os.path.join(config.RESULTS_DIR, "forecasts.json")
    comparison_path = os.path.join(config.RESULTS_DIR, "model_comparison.json")
    models_path = os.path.join(config.MODELS_DIR, "trained_models.pkl")
    data_path = os.path.join(config.ARTIFACTS_DIR, "prepared_data.pkl")

    if os.path.exists(forecasts_path):
        with open(forecasts_path, "r") as f:
            _forecasts = json.load(f)
        logger.info(f"Loaded forecasts for {len(_forecasts)} states")

    if os.path.exists(comparison_path):
        with open(comparison_path, "r") as f:
            _comparison = json.load(f)
        logger.info("Loaded model comparison results")

    if os.path.exists(models_path):
        _models_data = joblib.load(models_path)
        logger.info("Loaded trained models")

    if os.path.exists(data_path):
        _prepared_data = joblib.load(data_path)
        logger.info("Loaded prepared data")


# ─── Helper ──────────────────────────────────────────────
def _get_available_states():
    """Return list of available states."""
    if _forecasts:
        return sorted(_forecasts.keys())
    if _prepared_data is not None:
        return sorted(_prepared_data[config.STATE_COL].unique().tolist())
    return []


# ─── Routes ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """API documentation / health check."""
    return jsonify({
        "service": "Time Series Forecasting API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "GET /states": "List all available states",
            "GET /forecast/<state>": "Get 8-week forecast for a specific state",
            "GET /forecast/all": "Get forecasts for all states",
            "GET /models/comparison": "Get model comparison results across all states",
            "GET /models/best": "Get best model for each state",
            "POST /predict": "Generate on-demand predictions (JSON body: {state, horizon})",
            "GET /historical/<state>": "Get historical data for a state",
            "GET /dashboard": "Open the forecasting dashboard UI",
        },
    })


@app.route("/dashboard", methods=["GET"])
@app.route("/ui", methods=["GET"])
def dashboard():
    """Serve the browser dashboard."""
    return send_from_directory(UI_DIR, "index.html")


@app.route("/dashboard/<path:path>", methods=["GET"])
def dashboard_assets(path):
    """Serve dashboard static assets."""
    return send_from_directory(UI_DIR, path)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": _models_data is not None,
        "forecasts_available": _forecasts is not None,
        "states_count": len(_get_available_states()),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/states", methods=["GET"])
def list_states():
    """List all available states."""
    states = _get_available_states()
    return jsonify({
        "count": len(states),
        "states": states,
    })


@app.route("/forecast/<state>", methods=["GET"])
def get_forecast(state):
    """Get the pre-computed 8-week forecast for a specific state."""
    if _forecasts is None:
        return jsonify({"error": "No forecasts available. Run the training pipeline first."}), 503

    # Case-insensitive lookup
    state_key = None
    for k in _forecasts:
        if k.lower() == state.lower():
            state_key = k
            break

    if state_key is None:
        return jsonify({
            "error": f"State '{state}' not found",
            "available_states": _get_available_states(),
        }), 404

    forecast = _forecasts[state_key]
    return jsonify({
        "state": state_key,
        "forecast_horizon_weeks": len(forecast["dates"]),
        "best_model": forecast["model"],
        "predictions": [
            {"date": d, "predicted_sales": v}
            for d, v in zip(forecast["dates"], forecast["values"])
        ],
    })


@app.route("/forecast/all", methods=["GET"])
def get_all_forecasts():
    """Get forecasts for all states."""
    if _forecasts is None:
        return jsonify({"error": "No forecasts available. Run the training pipeline first."}), 503

    return jsonify({
        "count": len(_forecasts),
        "forecasts": {
            state: {
                "best_model": data["model"],
                "predictions": [
                    {"date": d, "predicted_sales": v}
                    for d, v in zip(data["dates"], data["values"])
                ],
            }
            for state, data in _forecasts.items()
        },
    })


@app.route("/models/comparison", methods=["GET"])
def model_comparison():
    """Get detailed model comparison results."""
    if _comparison is None:
        return jsonify({"error": "No comparison results available."}), 503

    return jsonify(_comparison)


@app.route("/models/best", methods=["GET"])
def best_models():
    """Get the best model selected for each state."""
    if _comparison is None:
        return jsonify({"error": "No comparison results available."}), 503

    return jsonify({
        "primary_metric": _comparison.get("primary_metric", config.PRIMARY_METRIC),
        "best_models": _comparison.get("best_models", {}),
    })


@app.route("/predict", methods=["POST"])
def predict_on_demand():
    """
    Generate predictions on-demand using trained models.
    
    Request body (JSON):
    {
        "state": "California",
        "horizon": 8  (optional, default 8)
    }
    """
    if _models_data is None:
        return jsonify({"error": "No trained models available. Run the pipeline first."}), 503

    data = request.get_json()
    if not data or "state" not in data:
        return jsonify({"error": "Request body must include 'state'"}), 400

    state = data["state"]
    horizon = data.get("horizon", config.FORECAST_HORIZON)

    # Validate
    if horizon < 1 or horizon > 52:
        return jsonify({"error": "Horizon must be between 1 and 52 weeks"}), 400

    # Find state (case-insensitive)
    state_key = None
    for k in _models_data.get("trained_models", {}):
        if k.lower() == state.lower():
            state_key = k
            break

    if state_key is None:
        return jsonify({
            "error": f"No trained model for state '{state}'",
            "available_states": list(_models_data.get("best_models", {}).keys()),
        }), 404

    try:
        model_info = _models_data["trained_models"][state_key]
        model = model_info["model_instance"]

        # Get state data for context
        state_data = None
        if _prepared_data is not None:
            state_data = _prepared_data[_prepared_data[config.STATE_COL] == state_key].copy()
            state_data = state_data.sort_values(config.DATE_COL).reset_index(drop=True)

        preds = model.predict(horizon=horizon, last_known_data=state_data)

        # Generate dates
        if state_data is not None:
            last_date = state_data[config.DATE_COL].max()
        else:
            last_date = pd.Timestamp.now()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq=config.FREQ,
        )

        return jsonify({
            "state": state_key,
            "model_used": model_info["model_name"],
            "horizon_weeks": horizon,
            "predictions": [
                {"date": d.strftime("%Y-%m-%d"), "predicted_sales": round(float(v), 2)}
                for d, v in zip(future_dates, preds)
            ],
        })

    except Exception as e:
        logger.error(f"Prediction error for {state}: {traceback.format_exc()}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/historical/<state>", methods=["GET"])
def get_historical(state):
    """Get historical data for a state."""
    if _prepared_data is None:
        return jsonify({"error": "No data available."}), 503

    # Case-insensitive
    state_key = None
    for s in _prepared_data[config.STATE_COL].unique():
        if s.lower() == state.lower():
            state_key = s
            break

    if state_key is None:
        return jsonify({"error": f"State '{state}' not found"}), 404

    state_df = _prepared_data[_prepared_data[config.STATE_COL] == state_key].sort_values(config.DATE_COL)

    # Optional: limit rows
    limit = request.args.get("limit", default=None, type=int)
    if limit:
        state_df = state_df.tail(limit)

    return jsonify({
        "state": state_key,
        "count": len(state_df),
        "data": [
            {
                "date": row[config.DATE_COL].strftime("%Y-%m-%d"),
                "sales": round(float(row[config.TARGET_COL]), 2),
            }
            for _, row in state_df.iterrows()
        ],
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "message": str(e)}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


# ─── Startup ─────────────────────────────────────────────
def create_app():
    """Application factory."""
    from src.utils import setup_logging
    setup_logging()
    _load_artifacts()
    return app


if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging()
    _load_artifacts()
    logger.info(f"Starting API server on {config.API_HOST}:{config.API_PORT}")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG)
