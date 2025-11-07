"""
AQI Forecast (24/48/72h) — Streamlit UI

Run:
    streamlit run app/streamlit_app.py
"""

import os
import json
from datetime import timezone
from zoneinfo import ZoneInfo

import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# CONFIG
CITY = os.getenv("CITY", "Karachi")
MODELS_DIR = "models/latest"
PKT = ZoneInfo("Asia/Karachi")
UTC = timezone.utc


# CATEGORY
def pm25_category(value: float):
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


def category_badge(cat: str):
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


# CACHED LOADERS
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


# DATA HELPERS
def geocode(city: str):

    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1}, timeout=20)
    r.raise_for_status()
    j = r.json()
    first = j["results"][0]
    return first["latitude"], first["longitude"]


def fetch_forecast_weather(lat: float, lon: float):
    
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


# FEATURE BUILDER
def build_current_feature_row(feature_names, pm25_hist: pd.DataFrame, weather_row: pd.Series):
    # we need at least 24h of PM2.5 to compute lag24, ma24
    pm25_hist = pm25_hist.sort_values("ts").copy()
    if len(pm25_hist) < 24:
        raise ValueError("Need at least 24 hourly PM2.5 rows to build features.")

    # features
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

    # order features
    X = np.array([[row_dict[c] for c in feature_names]], dtype=float)
    return X, ts_utc


# SHAP
def try_shap(model, feature_names, X_row):
    try:
        import shap

        # tree models
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_row)
            values = values[0] if isinstance(values, list) else values
        else:
            # linear model (Ridge)
            background = np.zeros((10, len(feature_names)))
            explainer = shap.LinearExplainer(model, background)
            values = explainer.shap_values(X_row)

        return feature_names, np.array(values).flatten()
    except Exception:
        return None, None


# STREAMLIT UI
st.set_page_config(page_title="AQI Forecast", page_icon="🌫️", layout="wide")
st.title("🌫️ AQI Forecast — 24h / 48h / 72h")
st.caption(f"City: **{CITY}** — times shown in **Asia/Karachi (PKT)**")

# 1) load metadata + models
feature_names, report = load_meta()
models = load_models()

if not models:
    st.error("No models found in models/latest/. Run training first.")
    st.stop()

# Winners
winners = report.get("horizons", {})
st.markdown("**Trained model winners (latest run):**")
st.write({
    "+24h": winners.get("h24", {}).get("best_model", "n/a"),
    "+48h": winners.get("h48", {}).get("best_model", "n/a"),
    "+72h": winners.get("h72", {}).get("best_model", "n/a"),
})

# 2) load weather + pm25
lat, lon = geocode(CITY)
forecast_df = fetch_forecast_weather(lat, lon)
pm25_hist = load_recent_pm25(CITY)

if pm25_hist.empty:
    st.warning(f"No PM2.5 history found at data/raw/pm25_{CITY}.parquet. Run ingest first.")
    st.stop()

# 3) make current feature row
current_weather = forecast_df.iloc[0]
X_row, ts_utc = build_current_feature_row(feature_names, pm25_hist, current_weather)

ts_pkt = ts_utc.astimezone(PKT).strftime("%Y-%m-%d %H:%M")
st.caption(f"Prediction reference time: **{ts_pkt} PKT**")

# 4) show forecasts
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

# 5) SHAP: let user pick which horizon to explain
horizon_to_explain = st.selectbox(
    "Explain which forecast?",
    options=[24, 48, 72],
    format_func=lambda h: f"+{h}h",
    index=0,
)

model_to_explain = models.get(horizon_to_explain)
if model_to_explain:
    labels, values = try_shap(model_to_explain, feature_names, X_row)
    if labels is not None:
        st.subheader(f"Why the +{horizon_to_explain}h prediction?")
        expl_df = (
            pd.DataFrame({"feature": labels, "influence": values})
            .sort_values("influence", ascending=False)
        )
        st.dataframe(expl_df, width="stretch")
        st.bar_chart(expl_df.set_index("feature"))
    else:
        st.info("SHAP could not be computed for this model.")
else:
    st.info("No model found for that horizon.")

# 6) debug
with st.expander("Recent PM2.5 (UTC)"):
    st.dataframe(pm25_hist.tail(10), width="stretch")

with st.expander("Training report (JSON)"):
    st.json(report)
