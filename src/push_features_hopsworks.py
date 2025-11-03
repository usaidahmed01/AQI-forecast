# """
# push_features_hopsworks.py
# Upload engineered features (data/features/features.parquet) to Hopsworks Feature Store.

# Env:
#   HOPSWORKS_PROJECT
#   HOPSWORKS_API_KEY
# """

# import os
# import pandas as pd
# import hopsworks
# from zoneinfo import ZoneInfo

# FEATURES_PATH = "data/features/features.parquet"
# FG_NAME = "aqi_features_hourly"
# FG_VERSION = 1  # bump when schema changes

# def main():
#     if not os.path.exists(FEATURES_PATH) or os.path.getsize(FEATURES_PATH) == 0:
#         raise FileNotFoundError(f"{FEATURES_PATH} missing or empty. Run features first.")
#     df = pd.read_parquet(FEATURES_PATH)

#     # Normalize event time to UTC for the feature store
#     df["ts"] = pd.to_datetime(df["ts"], utc=True)
#     if df["ts"].dt.tz is None:
#         # if somehow naive, assume Asia/Karachi then convert
#         df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(ZoneInfo("Asia/Karachi")).dt.tz_convert("UTC")

#     project = hopsworks.login(
#         project=os.getenv("HOPSWORKS_PROJECT"),
#         api_key_value=os.getenv("HOPSWORKS_API_KEY"),
#     )
#     fs = project.get_feature_store()

#     fg = fs.get_or_create_feature_group(
#         name=FG_NAME,
#         version=FG_VERSION,
#         primary_key=["ts"],
#         event_time="ts",
#         description="Engineered hourly AQI features (lags/rolling + weather/time).",
#         online_enabled=False,
#     )

#     fg.insert(df, write_options={"wait_for_job": True})
#     print(f"Inserted {len(df)} rows into Feature Group {FG_NAME} v{FG_VERSION}.")

# if __name__ == "__main__":
#     main()
















"""
push_features_hopsworks.py
Upload engineered features (data/features/features.parquet) to Hopsworks Feature Store.

Env (GitHub Secrets -> env in workflow step):
  HOPSWORKS_PROJECT
  HOPSWORKS_API_KEY
"""

import os
import pandas as pd
from zoneinfo import ZoneInfo

# HSFS: use the connection() FUNCTION in rc7 (not connection.connection / not Connection class)
from hsfs import connection as hsfs_connection

FEATURES_PATH = "data/features/features.parquet"
FG_NAME = "aqi_features_hourly"
FG_VERSION = 1  # bump if schema changes

def main():
    if not os.path.exists(FEATURES_PATH) or os.path.getsize(FEATURES_PATH) == 0:
        raise FileNotFoundError(f"{FEATURES_PATH} missing or empty. Run features first.")

    df = pd.read_parquet(FEATURES_PATH)

    # Normalize event time to UTC
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # If somehow naive, assume Asia/Karachi then convert to UTC
    if getattr(df["ts"].dt, "tz", None) is None:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(ZoneInfo("Asia/Karachi")).dt.tz_convert("UTC")

    project = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not project or not api_key:
        raise RuntimeError("HOPSWORKS_PROJECT / HOPSWORKS_API_KEY not set.")

    # ✅ Call the function directly
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
        # Be nice and close
        try:
            conn.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
