"""
push_features_hopsworks.py
Upload engineered features (data/features/features.parquet) to Hopsworks Feature Store.

Env (GitHub Secrets -> env in workflow step):
  HOPSWORKS_PROJECT
  HOPSWORKS_API_KEY
"""

import os
import pandas as pd
import hopsworks
from zoneinfo import ZoneInfo

FEATURES_PATH = "data/features/features.parquet"
FG_NAME = "aqi_features_hourly"
FG_VERSION = 1  # bump when schema changes

def main():
    # ---- load features ----
    if not os.path.exists(FEATURES_PATH) or os.path.getsize(FEATURES_PATH) == 0:
        raise FileNotFoundError(f"{FEATURES_PATH} missing or empty. Run features first.")
    df = pd.read_parquet(FEATURES_PATH)

    # normalize time to UTC
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    if df["ts"].dt.tz is None or str(df["ts"].dt.tz.iloc[0]) == "None":
        # assume Asia/Karachi if naive then convert to UTC
        df["ts"] = df["ts"].dt.tz_localize(ZoneInfo("Asia/Karachi")).dt.tz_convert("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")

    # ---- login & get store ----
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not project_name or not api_key:
        raise RuntimeError("HOPSWORKS_PROJECT / HOPSWORKS_API_KEY not set in env.")

    project = hopsworks.login(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()  # default FS for the project

    # ---- create/get feature group ----
    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        primary_key=["ts"],
        event_time="ts",
        description="Engineered hourly AQI features (lags/rolling + weather/time).",
        online_enabled=False,
    )

    # ---- insert (synchronous) ----
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted {len(df)} rows into Feature Group {FG_NAME} v{FG_VERSION}.")

if __name__ == "__main__":
    main()
