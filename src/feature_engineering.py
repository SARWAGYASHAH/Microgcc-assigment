"""
Feature Engineering Module
Creates lag features, rolling statistics, calendar features, and holiday flags.
"""
import pandas as pd
import numpy as np
import holidays
import logging

import config

logger = logging.getLogger(__name__)


def add_lag_features(df: pd.DataFrame, target: str = None, lags: list = None) -> pd.DataFrame:
    """Add lag features, including the required t-1, t-7, and t-30 periods."""
    target = target or config.TARGET_COL
    lags = lags or config.LAG_WEEKS

    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
        logger.debug(f"Added lag_{lag}")

    return df


def add_rolling_features(df: pd.DataFrame, target: str = None, windows: list = None) -> pd.DataFrame:
    """Add rolling mean and standard deviation features."""
    target = target or config.TARGET_COL
    windows = windows or config.ROLLING_WINDOWS

    for w in windows:
        df[f"rolling_mean_{w}"] = df[target].shift(1).rolling(window=w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = df[target].shift(1).rolling(window=w, min_periods=1).std()
        # Fill std NaN (from single-value windows) with 0
        df[f"rolling_std_{w}"] = df[f"rolling_std_{w}"].fillna(0)
        logger.debug(f"Added rolling_mean_{w} and rolling_std_{w}")

    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    """Add calendar-based features from the date column."""
    date_col = date_col or config.DATE_COL

    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    df["day_of_week"] = df[date_col].dt.dayofweek  # 0=Monday
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
    df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)

    # Cyclical encoding for seasonality (helps ML models)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    logger.debug("Added calendar features")
    return df


def add_holiday_features(df: pd.DataFrame, date_col: str = None, country: str = None) -> pd.DataFrame:
    """Add US holiday flags."""
    date_col = date_col or config.DATE_COL
    country = country or config.COUNTRY

    # Get all years in the data
    years = df[date_col].dt.year.unique().tolist()
    # Extend by 1 year for forecasting
    years.append(max(years) + 1)

    us_holidays = holidays.country_holidays(country, years=years)

    # Check if any day in the week is a holiday
    df["is_holiday"] = df[date_col].apply(
        lambda d: int(any((d + pd.Timedelta(days=i)) in us_holidays for i in range(7)))
    )

    # Holiday proximity: days until next major holiday (simplified)
    df["holiday_flag"] = df["is_holiday"]

    logger.debug("Added holiday features")
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-index based trend features."""
    df["time_index"] = np.arange(len(df))
    df["time_index_norm"] = df["time_index"] / len(df)  # Normalized 0-1
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline for a SINGLE STATE's data.
    Input must be sorted by date.
    """
    df = df.copy()
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_trend_features(df)

    logger.info(f"Feature engineering complete. Columns: {len(df.columns)}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature column names (excludes target, date, state, category)."""
    exclude = {config.TARGET_COL, config.DATE_COL, config.STATE_COL, config.CATEGORY_COL}
    return [c for c in df.columns if c not in exclude]


def prepare_ml_data(df: pd.DataFrame, drop_na: bool = True):
    """
    Prepare DataFrame for ML models (XGBoost).
    Returns X (features) and y (target) with NaN rows from lag features dropped.
    """
    feat_cols = get_feature_columns(df)
    ml_df = df[[config.TARGET_COL] + feat_cols].copy()

    if drop_na:
        ml_df = ml_df.dropna()

    X = ml_df[feat_cols]
    y = ml_df[config.TARGET_COL]

    return X, y
