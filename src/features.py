import os, pandas as pd, numpy as np

def add_time_features(df, col="ts"):
    df[col] = pd.to_datetime(df[col], utc=True)
    df["hour"] = df[col].dt.hour
    df["dow"]  = df[col].dt.dayofweek
    df["dom"]  = df[col].dt.day
    df["month"]= df[col].dt.month
    return df

def add_lag_roll_targets(df, target="pm25"):
    # Lags: last hour and yesterday same hour (24h)
    df[f"{target}_lag1"]  = df[target].shift(1)
    df[f"{target}_lag24"] = df[target].shift(24)
    # Rolling means for smoothing
    df[f"{target}_ma6"]   = df[target].rolling(6).mean()
    df[f"{target}_ma24"]  = df[target].rolling(24).mean()
    # Change rate: (current - last hour)
    df[f"{target}_chg1"]  = df[target] - df[f"{target}_lag1"]
    return df

def make_supervised(df, target="pm25"):
    """
    Build training rows for next 3 days (72 hours).
    We'll create 3 targets: t+24, t+48, t+72 hours.
    """
    out = df.copy()
    for h in [24, 48, 72]:
        out[f"{target}_tplus_{h}"] = out[target].shift(-h)
    # Drop rows with NaNs from lags/future shifts
    out = out.dropna().reset_index(drop=True)
    return out

def build_features(in_path="data/raw/merged_latest.parquet", out_path="data/features/features.parquet"):
    if (not os.path.exists(in_path)) or os.path.getsize(in_path) == 0:
        raise FileNotFoundError(
            f"{in_path} is missing or empty. Run `python src/ingest.py` again "
            "after fixing ingestion (increase OpenAQ radius or enable fallback)."
        )
    df = pd.read_parquet(in_path)
    if df.empty:
        raise ValueError("Merged raw dataset has 0 rows. Fix ingestion first.")

    df = df.sort_values("ts")

    # ==== your existing helpers here ====
    # add_time_features, add_lag_roll_targets, make_supervised

    df = add_time_features(df, "ts")
    df = add_lag_roll_targets(df, "pm25")
    df = make_supervised(df, "pm25")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path)
    return df

if __name__ == "__main__":
    df = build_features()
    print(df.head())
