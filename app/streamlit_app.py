# # app/streamlit_app.py
# """
# AQI Forecast (24/48/72h) — Streamlit UI

# - Loads best models from models/latest/
# - Reads your latest PM2.5 history from data/raw/pm25_<CITY>.parquet
# - Pulls 3-day hourly weather forecast from Open-Meteo (free, no key)
# - Builds the "now-ish" feature row just like features.py (lags/rolling/time)
# - Predicts t+24 / t+48 / t+72 and shows quick alert categories
# - Optional SHAP explanation for +24h model
# - Shows all reference timestamps in *Asia/Karachi (PKT)* to avoid confusion

# Run locally:
#   streamlit run app/streamlit_app.py
# """

# import os, json, joblib, requests, pandas as pd, numpy as np
# import streamlit as st
# from datetime import datetime, timedelta, timezone
# from zoneinfo import ZoneInfo  # Python 3.9+
# from math import sqrt

# # ------------------------- CONFIG -------------------------
# CITY = os.getenv("CITY", "Karachi")
# MODELS_DIR = "models/latest"
# PKT = ZoneInfo("Asia/Karachi")     # display times in Pakistan time
# UTC = timezone.utc                 # internal math in UTC

# # -------------------- ALERT CATEGORIES --------------------
# # US EPA PM2.5 thresholds (µg/m³)
# def pm25_category(x: float) -> str:
#     if x <= 12.0:   return "Good"
#     if x <= 35.4:   return "Moderate"
#     if x <= 55.4:   return "Unhealthy for Sensitive"
#     if x <= 150.4:  return "Unhealthy"
#     if x <= 250.4:  return "Very Unhealthy"
#     return "Hazardous"

# # Small helper to colorize category
# def category_badge(cat: str) -> str:
#     colors = {
#         "Good": "#22c55e",
#         "Moderate": "#eab308",
#         "Unhealthy for Sensitive": "#f97316",
#         "Unhealthy": "#ef4444",
#         "Very Unhealthy": "#8b5cf6",
#         "Hazardous": "#7f1d1d",
#     }
#     c = colors.get(cat, "#64748b")
#     return f"<span style='background:{c};color:white;padding:4px 8px;border-radius:999px;font-size:12px'>{cat}</span>"

# # ---------------------- CACHED LOADERS --------------------
# @st.cache_data
# def load_meta():
#     with open(os.path.join(MODELS_DIR, "features.json")) as f:
#         feats = json.load(f)
#     with open(os.path.join(MODELS_DIR, "report.json")) as f:
#         rpt = json.load(f)
#     return feats["feature_names"], rpt

# @st.cache_data
# def load_models():
#     models = {}
#     for h in [24, 48, 72]:
#         cands = [p for p in os.listdir(MODELS_DIR) if p.endswith(f"tplus{h}.joblib")]
#         if cands:
#             models[h] = joblib.load(os.path.join(MODELS_DIR, cands[0]))
#     return models

# @st.cache_data
# def load_recent_pm25(city=CITY):
#     path = f"data/raw/pm25_{city}.parquet"
#     if not os.path.exists(path):
#         return pd.DataFrame(columns=["ts", "pm25"])
#     df = pd.read_parquet(path, engine="pyarrow")
#     # ensure UTC for all time math
#     df["ts"] = pd.to_datetime(df["ts"], utc=True)
#     return df.sort_values("ts")

# # ------------------- DATA FETCH HELPERS -------------------
# def fetch_forecast_weather(lat, lon):
#     """
#     Get next 3-day hourly weather in UTC. We’ll display PKT in the UI.
#     """
#     url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude": lat, "longitude": lon,
#         "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure",
#         "forecast_days": 3,
#         "timezone": "UTC"
#     }
#     r = requests.get(url, params=params, timeout=30)
#     r.raise_for_status()
#     h = r.json()["hourly"]
#     df = pd.DataFrame(h).rename(columns={"time":"ts"})
#     df["ts"] = pd.to_datetime(df["ts"], utc=True)
#     return df

# def geocode(city):
#     r = requests.get(
#         "https://geocoding-api.open-meteo.com/v1/search",
#         params={"name": city, "count": 1}, timeout=20
#     )
#     r.raise_for_status()
#     j = r.json()
#     first = j["results"][0]
#     return first["latitude"], first["longitude"]

# # --------------- FEATURE ROW (like features.py) ---------------
# def build_current_feature_row(feature_names, history_pm25: pd.DataFrame, weather_now_row: pd.Series):
#     """
#     Create the 1-row feature vector for 'now-ish' using recent PM2.5 history + current weather.
#     Mirrors features.py: lags, rolling means, and time features.
#     """
#     df = history_pm25.copy().sort_values("ts")

#     # Must have at least 24h history to compute lag24/ma24 cleanly
#     if len(df) < 24:
#         raise ValueError("Not enough PM2.5 history to compute 24h features yet. Need >= 24 rows.")

#     df["pm25_lag1"]  = df["pm25"].shift(1)
#     df["pm25_lag24"] = df["pm25"].shift(24)
#     df["pm25_ma6"]   = df["pm25"].rolling(6).mean()
#     df["pm25_ma24"]  = df["pm25"].rolling(24).mean()
#     df["pm25_chg1"]  = df["pm25"] - df["pm25_lag1"]

#     last = df.dropna().iloc[-1] if not df.dropna().empty else None
#     if last is None:
#         raise ValueError("Recent history available, but lags/rolls are NaN. Wait for more data.")

#     # time features from weather 'now' timestamp
#     ts_utc = pd.to_datetime(weather_now_row["ts"], utc=True)
#     hour, dow, dom, month = ts_utc.hour, ts_utc.dayofweek, ts_utc.day, ts_utc.month

#     # Build feature dict exactly in training-schema
#     row = {
#         "pm25": float(last["pm25"]),
#         "pm25_lag1": float(last["pm25_lag1"]),
#         "pm25_lag24": float(last["pm25_lag24"]),
#         "pm25_ma6": float(last["pm25_ma6"]),
#         "pm25_ma24": float(last["pm25_ma24"]),
#         "pm25_chg1": float(last["pm25_chg1"]),
#         "temperature_2m": float(weather_now_row["temperature_2m"]),
#         "relative_humidity_2m": float(weather_now_row["relative_humidity_2m"]),
#         "wind_speed_10m": float(weather_now_row["wind_speed_10m"]),
#         "surface_pressure": float(weather_now_row["surface_pressure"]),
#         "hour": hour, "dow": dow, "dom": dom, "month": month
#     }

#     # Order columns to match training exactly
#     X = np.array([[row[c] for c in feature_names]], dtype=float)
#     return X, ts_utc

# # ------------------------- SHAP HELPERS -------------------------
# def try_shap(model, feature_names, X_row):
#     """
#     Return (labels, values) for a simple SHAP bar if model is RF or Ridge.
#     We handle both tree and linear explainers; fall back gracefully if not available.
#     """
#     try:
#         import shap
#         # Tree-based?
#         if hasattr(model, "estimators_"):
#             explainer = shap.TreeExplainer(model)
#             vals = explainer.shap_values(X_row)
#             # TreeExplainer returns list per class for classifiers; for regressor it's array
#             vals = vals[0] if isinstance(vals, list) else vals
#         else:
#             # Linear model (Ridge)
#             background = np.zeros((10, len(feature_names)))
#             explainer = shap.LinearExplainer(model, background, feature_dependence="independent")
#             vals = explainer.shap_values(X_row)
#         return feature_names, np.array(vals).flatten()
#     except Exception:
#         return None, None

# # ============================= UI =============================
# st.set_page_config(page_title="AQI Forecast", page_icon="🌫️", layout="wide")
# st.title("🌫️ AQI Forecast — 24h / 48h / 72h")
# st.caption(f"City: **{CITY}** — all times shown in **Asia/Karachi (PKT)**")

# # Load metadata + models
# feature_names, report = load_meta()
# models = load_models()
# if not models:
#     st.error("No models found in models/latest/. Please train and copy the latest models.")
#     st.stop()

# # Inputs (geocode + forecast) and recent PM2.5
# lat, lon = geocode(CITY)
# forecast = fetch_forecast_weather(lat, lon)      # UTC
# pm25_hist = load_recent_pm25(CITY)               # UTC

# if pm25_hist.empty:
#     st.warning(f"No PM2.5 history at data/raw/pm25_{CITY}.parquet — run ingest first.")
#     st.stop()

# # Use the first upcoming forecast hour as "now-ish"
# now_row = forecast.iloc[0]
# X_row, ts_utc = build_current_feature_row(feature_names, pm25_hist, now_row)
# ts_pkt_str = ts_utc.astimezone(PKT).strftime("%Y-%m-%d %H:%M")
# st.caption(f"Prediction reference time: **{ts_pkt_str} PKT** (source data in UTC underneath)")

# # Predictions + alert badges
# cols = st.columns(3)
# for i, h in enumerate([24, 48, 72]):
#     m = models.get(h)
#     if m is None:
#         cols[i].error(f"No model for +{h}h")
#         continue
#     yhat = float(m.predict(X_row)[0])
#     cat = pm25_category(yhat)
#     cols[i].metric(label=f"PM2.5 forecast (+{h}h)", value=f"{yhat:.1f} µg/m³")
#     cols[i].markdown(category_badge(cat), unsafe_allow_html=True)

# # SHAP for +24h model (optional explainability)
# m24 = models.get(24)
# if m24:
#     labels, vals = try_shap(m24, feature_names, X_row)
#     if labels is not None:
#         st.subheader("Why the +24h prediction? (SHAP)")
#         expl = pd.DataFrame({"feature": labels, "influence": vals}).sort_values(
#             "influence", ascending=False
#         )
#         st.dataframe(expl, use_container_width=True)
#         # quick bar chart
#         st.bar_chart(expl.set_index("feature"))
#     else:
#         st.info("SHAP explanation not available for this model/runtime.")

# # Small debug/peek expanders (optional)
# with st.expander("See last few PM2.5 rows (UTC)"):
#     st.dataframe(pm25_hist.tail(10), use_container_width=True)
# with st.expander("Model report (from training)"):
#     st.json(report)




























# app/streamlit_app.py
"""
AQI Forecast (24/48/72h) — Streamlit UI

Run:
    streamlit run app/streamlit_app.py
"""

import os
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# --------------------- BASIC CONFIG ---------------------
CITY = os.getenv("CITY", "Karachi")
MODELS_DIR = "models/latest"
PKT = ZoneInfo("Asia/Karachi")
UTC = timezone.utc


# --------------------- CATEGORY HELPERS -----------------
def pm25_category(value: float) -> str:
    if value <= 12.0:
        return "Good"
    if value <= 35.4:
        return "Moderate"
    if value <= 55.4:
        return "Unhealthy for Sensitive"
    if value <= 150.4:
        return "Unhealthy"
    if value <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


def category_badge(cat: str) -> str:
    colors = {
        "Good": "#22c55e",
        "Moderate": "#eab308",
        "Unhealthy for Sensitive": "#f97316",
        "Unhealthy": "#ef4444",
        "Very Unhealthy": "#8b5cf6",
        "Hazardous": "#7f1d1d",
    }
    bg = colors.get(cat, "#64748b")
    return (
        f"<span style='background:{bg};color:white;"
        "padding:4px 8px;border-radius:999px;font-size:12px'>"
        f"{cat}</span>"
    )


# --------------------- CACHED LOADERS -------------------
@st.cache_data
def load_meta():
    with open(os.path.join(MODELS_DIR, "features.json")) as f:
        feats = json.load(f)
    with open(os.path.join(MODELS_DIR, "report.json")) as f:
        rpt = json.load(f)
    return feats["feature_names"], rpt


@st.cache_data
def load_models():
    models = {}
    for h in [24, 48, 72]:
        # find any file that ends with tplus{h}.joblib
        files = [
            p for p in os.listdir(MODELS_DIR)
            if p.endswith(f"tplus{h}.joblib")
        ]
        if files:
            models[h] = joblib.load(os.path.join(MODELS_DIR, files[0]))
    return models


@st.cache_data
def load_recent_pm25(city: str):
    path = f"data/raw/pm25_{city}.parquet"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["ts", "pm25"])
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


# --------------------- REMOTE DATA HELPERS ---------------
def geocode(city: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1}, timeout=20)
    r.raise_for_status()
    j = r.json()
    first = j["results"][0]
    return first["latitude"], first["longitude"]


def fetch_forecast_weather(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,relative_humidity_2m,"
            "wind_speed_10m,surface_pressure"
        ),
        "forecast_days": 3,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


# --------------------- FEATURE BUILDER -------------------
def build_current_feature_row(feature_names, pm25_hist: pd.DataFrame, weather_row: pd.Series):
    # need at least 24 rows to build lag24 / ma24
    pm25_hist = pm25_hist.sort_values("ts").copy()
    if len(pm25_hist) < 24:
        raise ValueError("Need at least 24 hourly PM2.5 rows to build features.")

    pm25_hist["pm25_lag1"] = pm25_hist["pm25"].shift(1)
    pm25_hist["pm25_lag24"] = pm25_hist["pm25"].shift(24)
    pm25_hist["pm25_ma6"] = pm25_hist["pm25"].rolling(6).mean()
    pm25_hist["pm25_ma24"] = pm25_hist["pm25"].rolling(24).mean()
    pm25_hist["pm25_chg1"] = pm25_hist["pm25"] - pm25_hist["pm25_lag1"]

    last = pm25_hist.dropna().iloc[-1]

    ts_utc = pd.to_datetime(weather_row["ts"], utc=True)
    hour = ts_utc.hour
    dow = ts_utc.dayofweek
    dom = ts_utc.day
    month = ts_utc.month

    row_dict = {
        "pm25": float(last["pm25"]),
        "pm25_lag1": float(last["pm25_lag1"]),
        "pm25_lag24": float(last["pm25_lag24"]),
        "pm25_ma6": float(last["pm25_ma6"]),
        "pm25_ma24": float(last["pm25_ma24"]),
        "pm25_chg1": float(last["pm25_chg1"]),
        "temperature_2m": float(weather_row["temperature_2m"]),
        "relative_humidity_2m": float(weather_row["relative_humidity_2m"]),
        "wind_speed_10m": float(weather_row["wind_speed_10m"]),
        "surface_pressure": float(weather_row["surface_pressure"]),
        "hour": hour,
        "dow": dow,
        "dom": dom,
        "month": month,
    }

    X = np.array([[row_dict[c] for c in feature_names]], dtype=float)
    return X, ts_utc


# --------------------- OPTIONAL SHAP ---------------------
def try_shap(model, feature_names, X_row):
    try:
        import shap

        # tree model?
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_row)
            values = values[0] if isinstance(values, list) else values
        else:
            # linear model (ridge)
            background = np.zeros((10, len(feature_names)))
            explainer = shap.LinearExplainer(model, background)
            values = explainer.shap_values(X_row)

        return feature_names, np.array(values).flatten()
    except Exception:
        return None, None


# ======================= STREAMLIT UI =====================
st.set_page_config(page_title="AQI Forecast", page_icon="🌫️", layout="wide")
st.title("🌫️ AQI Forecast — 24h / 48h / 72h")
st.caption(f"City: **{CITY}** — times shown in **Asia/Karachi (PKT)**")

# load model assets
feature_names, report = load_meta()
models = load_models()

if not models:
    st.error("No models found in models/latest/. Run training first.")
    st.stop()

# load inputs (data + weather)
lat, lon = geocode(CITY)
forecast_df = fetch_forecast_weather(lat, lon)
pm25_hist = load_recent_pm25(CITY)

if pm25_hist.empty:
    st.warning(f"No PM2.5 history found at data/raw/pm25_{CITY}.parquet. Run ingest first.")
    st.stop()

# use the first forecast hour as the "current" reference
current_weather = forecast_df.iloc[0]
X_row, ts_utc = build_current_feature_row(feature_names, pm25_hist, current_weather)

ts_pkt = ts_utc.astimezone(PKT).strftime("%Y-%m-%d %H:%M")
st.caption(f"Prediction reference time: **{ts_pkt} PKT**")

# 3 columns for 24/48/72
cols = st.columns(3)
for i, h in enumerate([24, 48, 72]):
    model = models.get(h)
    if not model:
        cols[i].error(f"No model for +{h}h")
        continue

    pred = float(model.predict(X_row)[0])
    cat = pm25_category(pred)

    cols[i].metric(label=f"PM2.5 forecast (+{h}h)", value=f"{pred:.1f} µg/m³")
    cols[i].markdown(category_badge(cat), unsafe_allow_html=True)

# SHAP explain (only for 24h if model present)
m24 = models.get(24)
if m24:
    labels, values = try_shap(m24, feature_names, X_row)
    if labels is not None:
        st.subheader("Why the +24h prediction?")
        expl_df = pd.DataFrame({"feature": labels, "influence": values}).sort_values(
            "influence", ascending=False
        )
        st.dataframe(expl_df, use_container_width=True)
        st.bar_chart(expl_df.set_index("feature"))
    else:
        st.info("SHAP explanation not available in this environment.")

# debug boxes
with st.expander("Recent PM2.5 (UTC)"):
    st.dataframe(pm25_hist.tail(10), use_container_width=True)

with st.expander("Training report"):
    st.json(report)
