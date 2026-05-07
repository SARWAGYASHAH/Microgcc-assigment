"""
Main Pipeline Orchestrator
Loads data → engineers features → trains all models per state →
compares → selects best → generates 8-week forecasts → saves everything.
"""
import pandas as pd
import numpy as np
import logging
import json
import os
import time
import joblib

from src.data_loader import load_and_prepare, get_state_data, train_val_split
from src.feature_engineering import engineer_features
from src.model_selector import ModelSelector
from src.utils import setup_logging
import config

logger = logging.getLogger(__name__)


def run_pipeline(states: list = None):
    """
    Full training pipeline.
    
    Args:
        states: Optional list of states to process. If None, processes all states.
    """
    setup_logging()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  TIME SERIES FORECASTING PIPELINE")
    logger.info("=" * 70)

    # ── Step 1: Load and prepare data ──
    logger.info("\n[Step 1] Loading and preparing data...")
    df = load_and_prepare()

    all_states = sorted(df[config.STATE_COL].unique().tolist())
    process_states = states if states else all_states
    logger.info(f"Processing {len(process_states)} states: {process_states[:5]}{'...' if len(process_states) > 5 else ''}")

    # ── Step 2: Train & evaluate models per state ──
    logger.info("\n[Step 2] Training and evaluating models...")
    selector = ModelSelector()

    forecasts = {}  # {state: {dates: [...], values: [...], model: "..."}}

    for i, state in enumerate(process_states, 1):
        logger.info(f"\n[{i}/{len(process_states)}] State: {state}")

        state_df = get_state_data(df, state)
        train_df, val_df = train_val_split(state_df)

        # Train all models and get comparison
        selector.train_and_evaluate_state(state, train_df, val_df)

        # ── Step 3: Generate 8-week forecast with the best model ──
        best_model = selector.get_best_model(state)
        logger.info(f"Generating {config.FORECAST_HORIZON}-week forecast with {best_model.name}...")

        # For forecasting, retrain best model on ALL data (train + val)
        best_model.fit(state_df)
        future_preds = best_model.predict(horizon=config.FORECAST_HORIZON, last_known_data=state_df)

        # Generate future dates
        last_date = state_df[config.DATE_COL].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=config.FORECAST_HORIZON,
            freq=config.FREQ,
        )

        forecasts[state] = {
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "values": [round(float(v), 2) for v in future_preds],
            "model": best_model.name,
            "best_model_name": selector.best_models[state],
        }

        logger.info(f"Forecast for {state}: {future_preds[:3].round(0)}... (showing first 3)")

    # ── Step 4: Save results ──
    logger.info("\n[Step 4] Saving results...")

    # Save model comparison
    selector.save_results()

    # Save forecasts
    forecasts_path = os.path.join(config.RESULTS_DIR, "forecasts.json")
    with open(forecasts_path, "w") as f:
        json.dump(forecasts, f, indent=2)
    logger.info(f"Forecasts saved to {forecasts_path}")

    # Save the comparison table
    comparison_df = selector.get_comparison_table()
    comparison_path = os.path.join(config.RESULTS_DIR, "comparison_table.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Comparison table saved to {comparison_path}")

    # Save trained model objects for the API
    models_data = {
        "best_models": selector.best_models,
        "trained_models": {},
    }
    for state in process_states:
        best_name = selector.best_models[state]
        models_data["trained_models"][state] = {
            "model_name": best_name,
            "model_instance": selector.trained_models[state][best_name],
        }

    models_path = os.path.join(config.MODELS_DIR, "trained_models.pkl")
    joblib.dump(models_data, models_path)
    logger.info(f"Trained models saved to {models_path}")

    # Save the prepared data for API use
    data_path = os.path.join(config.ARTIFACTS_DIR, "prepared_data.pkl")
    joblib.dump(df, data_path)
    logger.info(f"Prepared data saved to {data_path}")

    # ── Summary ──
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"{'='*70}")

    # Print best model per state
    logger.info("\n[RESULTS] Best Model Summary:")
    logger.info(f"{'State':<20} {'Best Model':<15} {'MAPE (%)':<10}")
    logger.info("-" * 45)
    for state in process_states:
        best = selector.best_models[state]
        mape = selector.results[state][best]["metrics"]["mape"]
        logger.info(f"{state:<20} {best:<15} {mape:<10.2f}")

    return selector, forecasts


def run_quick_test(test_states: list = None):
    """Run the pipeline on a few states for quick testing."""
    test_states = test_states or ["California", "Texas", "New York"]
    return run_pipeline(states=test_states)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test mode
        selector, forecasts = run_quick_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "--states":
        # Specific states
        states = sys.argv[2:]
        selector, forecasts = run_pipeline(states=states)
    else:
        # Full pipeline
        selector, forecasts = run_pipeline()
