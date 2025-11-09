"""
Upload engineered features (data/features/features.parquet) to Hopsworks Feature Store.

Env (in GitHub Actions step):
  HOPSWORKS_PROJECT
  HOPSWORKS_API_KEY
"""

import os
from zoneinfo import ZoneInfo

import pandas as pd
from hsfs import connection as hsfs_connection

FEATURES_PATH = "data/features/features.parquet"
FG_NAME = "aqi_features_hourly"
FG_VERSION = 1  


def main():
    # 1) basic file check
    if (not os.path.exists(FEATURES_PATH)) or os.path.getsize(FEATURES_PATH) == 0:
        raise FileNotFoundError(f"{FEATURES_PATH} missing or empty. Run features first.")

    df = pd.read_parquet(FEATURES_PATH)

    # 2) normalize timestamp to UTC
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
    if "relative_humidity_2m" in df.columns:
        df["relative_humidity_2m"] = (
        pd.to_numeric(df["relative_humidity_2m"])
        .round(0)
        .astype("int64")
    )      

    # if somehow a naive ts sneaks in, assume Asia/Karachi and convert
    if getattr(df["ts"].dt, "tz", None) is None:
        df["ts"] = (
            pd.to_datetime(df["ts"])
            .dt.tz_localize(ZoneInfo("Asia/Karachi"))
            .dt.tz_convert("UTC")
        )

    project = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not project or not api_key:
        raise RuntimeError("HOPSWORKS_PROJECT / HOPSWORKS_API_KEY not set.")

    # 3) connect to Hopsworks
    conn = hsfs_connection(
        host="c.app.hopsworks.ai",
        port=443,
        project=project,
        api_key_value=api_key,
        engine="python",
    )

    try:
        fs = conn.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name=FG_NAME,
            version=FG_VERSION,
            primary_key=["ts"],
            event_time="ts",
            description="Engineered hourly AQI features (lags/rolling + weather/time).",
            online_enabled=False,
        )

        fg.insert(df, write_options={"wait_for_job": True})
        print(f"Inserted {len(df)} rows into Feature Group {FG_NAME} v{FG_VERSION}.")
    finally:
        try:
            conn.close()
        except Exception:
            pass



main()
