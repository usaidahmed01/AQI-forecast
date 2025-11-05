"""
Purpose:
  Take the merged raw data (PM2.5 + weather) and turn it into:
    - features the model can use (time, lags, rolling)
    - targets for future hours (t+24, t+48, t+72)

Input:
  data/raw/merged_latest.parquet
Output:
  data/features/features.parquet
"""

import os
import pandas as pd
import numpy as np  # kept since you imported it; fine to keep

# central names
TIME_COL = "ts"     # UTC timestamp
TARGET_COL = "pm25" # value we want to forecast


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic time columns:
      - hour of day
      - day of week
      - day of month
      - month
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)

    df["hour"] = df[TIME_COL].dt.hour
    df["dow"] = df[TIME_COL].dt.dayofweek
    df["dom"] = df[TIME_COL].dt.day
    df["month"] = df[TIME_COL].dt.month

    print("time featuring\n", df.head())
    return df


def add_lag_rolling_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PM2.5 history columns:
      - pm25_lag1   (1 hour ago)
      - pm25_lag24  (24 hours ago)
      - pm25_ma6    (6-hour moving avg)
      - pm25_ma24   (24-hour moving avg)
      - pm25_chg1   (current - 1 hour ago)
    """
    df = df.copy()

    df[f"{TARGET_COL}_lag1"] = df[TARGET_COL].shift(1)
    df[f"{TARGET_COL}_lag24"] = df[TARGET_COL].shift(24)
    df[f"{TARGET_COL}_ma6"] = df[TARGET_COL].rolling(6).mean()
    df[f"{TARGET_COL}_ma24"] = df[TARGET_COL].rolling(24).mean()
    df[f"{TARGET_COL}_chg1"] = df[TARGET_COL] - df[f"{TARGET_COL}_lag1"]

    print("Lag , Rolling\n", df.head())
    return df


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add future labels we want to predict:
      - pm25_tplus_24
      - pm25_tplus_48
      - pm25_tplus_72
    Shift is negative because we want "future" values on the current row.
    """
    df = df.copy()

    for h in [24, 48, 72]:
        col_name = f"{TARGET_COL}_tplus_{h}"
        df[col_name] = df[TARGET_COL].shift(-h)

    print("Future Targets\n", df.head())
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that have any NaN in features/targets.
    This happens near the edges (start of lags, end of future shifts).
    """
    cleaned = df.dropna().reset_index(drop=True)
    print("Clean Data\n", cleaned.head())
    return cleaned


def build_features(
    in_path: str = "data/raw/merged_latest.parquet",
    out_path: str = "data/features/features.parquet",
) -> pd.DataFrame:
    """
    Full feature pipeline:
      1) read merged parquet
      2) sort by time
      3) add time features
      4) add lag/rolling/change
      5) add future targets
      6) drop NaNs
      7) save to parquet
    """
    if (not os.path.exists(in_path)) or os.path.getsize(in_path) == 0:
        raise FileNotFoundError(f"{in_path} is missing or empty. Run ingest first.")

    df = pd.read_parquet(in_path)
    if df.empty:
        raise ValueError("Merged raw dataset has 0 rows. Fix ingestion first.")

    df = df.sort_values(TIME_COL)

    df = add_time_features(df)
    df = add_lag_rolling_change(df)
    df = add_future_targets(df)
    df = drop_incomplete_rows(df)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)

    return df


output = build_features()
print(output.head())
