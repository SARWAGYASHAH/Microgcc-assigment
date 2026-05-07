"""
Data Loader Module
Handles loading, cleaning, and resampling the raw Excel dataset.
"""
import pandas as pd
import numpy as np
import logging

import config

logger = logging.getLogger(__name__)


def load_raw_data(path: str = None) -> pd.DataFrame:
    """Load the raw Excel file and return a DataFrame."""
    path = path or config.DATA_PATH
    logger.info(f"Loading data from {path}")
    df = pd.read_excel(path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw data:
    - Parse dates (mixed formats)
    - Sort by state and date
    - Handle missing values
    - Resample to consistent weekly frequency per state
    """
    df = df.copy()

    # ── Parse dates ──
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], format="mixed", dayfirst=True)
    logger.info(f"Date range: {df[config.DATE_COL].min()} -> {df[config.DATE_COL].max()}")

    # ── Sort ──
    df = df.sort_values([config.STATE_COL, config.DATE_COL]).reset_index(drop=True)

    # ── Handle missing target values ──
    null_count = df[config.TARGET_COL].isnull().sum()
    if null_count > 0:
        logger.warning(f"Found {null_count} null values in '{config.TARGET_COL}', forward-filling per state")
        df[config.TARGET_COL] = df.groupby(config.STATE_COL)[config.TARGET_COL].transform(
            lambda s: s.fillna(method="ffill").fillna(method="bfill")
        )

    # ── Resample to regular weekly frequency per state ──
    resampled_frames = []
    for state, group in df.groupby(config.STATE_COL):
        group = group.set_index(config.DATE_COL)
        # Resample to weekly (Sunday end) and take the mean if duplicates exist
        weekly = group[[config.TARGET_COL]].resample(config.FREQ).mean()
        # Interpolate missing weeks (linear)
        weekly[config.TARGET_COL] = weekly[config.TARGET_COL].interpolate(method="linear")
        # Back-fill any remaining NaNs at the start
        weekly[config.TARGET_COL] = weekly[config.TARGET_COL].bfill()
        weekly[config.STATE_COL] = state
        weekly[config.CATEGORY_COL] = "Beverages"
        resampled_frames.append(weekly)

    result = pd.concat(resampled_frames).reset_index()
    result = result.rename(columns={"index": config.DATE_COL} if "index" in result.columns else {})

    logger.info(f"After resampling: {len(result)} rows, {result[config.STATE_COL].nunique()} states")
    logger.info(f"Weeks per state: ~{len(result) // result[config.STATE_COL].nunique()}")

    return result


def get_state_data(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Extract data for a single state, sorted by date."""
    state_df = df[df[config.STATE_COL] == state].copy()
    state_df = state_df.sort_values(config.DATE_COL).reset_index(drop=True)
    return state_df


def train_val_split(df: pd.DataFrame, val_weeks: int = None):
    """
    Split a single-state DataFrame into train and validation sets.
    Uses the last `val_weeks` weeks as validation (time-series aware, no leakage).
    """
    val_weeks = val_weeks or config.VALIDATION_WEEKS
    cutoff_idx = len(df) - val_weeks

    if cutoff_idx < 20:
        logger.warning(f"Very small training set ({cutoff_idx} rows). Results may be unreliable.")

    train = df.iloc[:cutoff_idx].copy()
    val = df.iloc[cutoff_idx:].copy()

    logger.info(f"Train: {len(train)} rows ({train[config.DATE_COL].min()} -> {train[config.DATE_COL].max()})")
    logger.info(f"Val:   {len(val)} rows ({val[config.DATE_COL].min()} -> {val[config.DATE_COL].max()})")

    return train, val


def load_and_prepare() -> pd.DataFrame:
    """Convenience function: load → clean → return."""
    raw = load_raw_data()
    clean = clean_data(raw)
    return clean
