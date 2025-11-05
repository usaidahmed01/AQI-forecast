# """
# features.py
# ===========
# Purpose:
#   Convert the merged raw table (PM2.5 + weather) into:
#     - INPUT FEATURES the model can learn from
#     - FUTURE TARGETS we want to predict (t+24, t+48, t+72 hours)
# Why:
#   Models need helpful signals (lags, rolling means, time-of-day, etc.).
#   We also need labels that represent the future values we’re trying to forecast.

# Input :
#   data/raw/merged_latest.parquet
# Output:
#   data/features/features.parquet
# """

# import os
# import pandas as pd
# import numpy as np

# # Central names so it’s easy to change later
# TIME_COL   = "ts"      # our canonical UTC timestamp column
# TARGET_COL = "pm25"    # what we want to predict in the future

# def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add calendar/time signals:
#       - hour of day       (0..23)
#       - day of week       (0=Monday..6=Sunday)
#       - day of month      (1..31)
#       - month of year     (1..12)
#     Reason:
#       Pollution patterns often follow daily/weekly cycles.
#     """
#     df = df.copy()
#     df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
#     df["hour"]  = df[TIME_COL].dt.hour
#     df["dow"]   = df[TIME_COL].dt.dayofweek
#     df["dom"]   = df[TIME_COL].dt.day
#     df["month"] = df[TIME_COL].dt.month
#     print("time featuring \n" , df.head())
#     return df

# def add_lag_rolling_change(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add "history" signals of PM2.5:
#       - lag1   : PM2.5 one hour ago
#       - lag24  : PM2.5 24 hours ago (same hour yesterday)
#       - ma6    : mean of last 6 hours (smooth short-term noise)
#       - ma24   : mean of last 24 hours (captures daily trend)
#       - chg1   : difference between current and last hour (is it rising/falling?)
#     Reason:
#       Near-future pollution is strongly related to recent history.
#     """
#     df = df.copy()
#     df[f"{TARGET_COL}_lag1"]  = df[TARGET_COL].shift(1)
#     df[f"{TARGET_COL}_lag24"] = df[TARGET_COL].shift(24)
#     df[f"{TARGET_COL}_ma6"]   = df[TARGET_COL].rolling(6).mean()
#     df[f"{TARGET_COL}_ma24"]  = df[TARGET_COL].rolling(24).mean()
#     df[f"{TARGET_COL}_chg1"]  = df[TARGET_COL] - df[f"{TARGET_COL}_lag1"]
#     print("Lag , Rolling \n" , df.head())
#     return df

# def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Create the "labels" we want to predict:
#       - pm25_tplus_24 : PM2.5 value 24 hours ahead
#       - pm25_tplus_48 : PM2.5 value 48 hours ahead
#       - pm25_tplus_72 : PM2.5 value 72 hours ahead
#     We shift NEGATIVELY because we want future rows aligned with current features.
#     """
#     df = df.copy()
#     for h in [24, 48, 72]:
#         df[f"{TARGET_COL}_tplus_{h}"] = df[TARGET_COL].shift(-h)
        
#     print("Future Targets \n" , df.head())
#     return df

# def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Any row that lacks any of the needed values (lags, rolling means, or future targets)
#     must be dropped because the model cannot train on NaNs.
#     """
    
#     print("Clean Data \n",  df.dropna().reset_index(drop=True).head())
#     return df.dropna().reset_index(drop=True)

# def build_features(
#     in_path: str = "data/raw/merged_latest.parquet",
#     out_path: str = "data/features/features.parquet",
# ) -> pd.DataFrame:
#     """
#     The full feature-engineering pipeline:
#       1) validate input exists and is not empty
#       2) add time features
#       3) add lag/rolling/change features
#       4) add future targets
#       5) drop rows with any missing values
#       6) save to Parquet for training
#     """
#     if (not os.path.exists(in_path)) or os.path.getsize(in_path) == 0:
#         raise FileNotFoundError(f"{in_path} is missing or empty. Run ingest first.")

#     df = pd.read_parquet(in_path)
#     if df.empty:
#         raise ValueError("Merged raw dataset has 0 rows. Fix ingestion first.")

#     df = df.sort_values(TIME_COL)

#     df = add_time_features(df)
#     df = add_lag_rolling_change(df)
#     df = add_future_targets(df)
#     df = drop_incomplete_rows(df)

#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     df.to_parquet(out_path)
#     return df

# if __name__ == "__main__":
#     out = build_features()
#     print(out.head())

























"""
features.py
===========
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


if __name__ == "__main__":
    out = build_features()
    print(out.head())
