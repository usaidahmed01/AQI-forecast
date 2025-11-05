"""
Purpose:
  1) Get a city's lat/lon and timezone (Open-Meteo Geocoding)
  2) Download hourly weather for a date range (Open-Meteo Forecast)
  3) Download hourly PM2.5 (OpenAQ v3); if none, fallback to Open-Meteo Air Quality
  4) Merge weather + PM2.5 on hourly timestamps
  5) Save raw snapshots to data/raw/

Outputs:
  - data/raw/pm25_<City>.parquet
  - data/raw/weather_<City>.parquet
  - data/raw/merged_latest.parquet

Before running (Windows PowerShell):
  $env:OPENAQ_API_KEY="YOUR_OPENAQ_KEY"
  $env:CITY="Karachi"
  $env:LOOKBACK_DAYS="7"
  python src/ingest.py
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

# ------------------------------- CONSTANTS ----------------------------------

OPENAQ_BASE = "https://api.openaq.org/v3"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]

MAX_RADIUS_M = 25_000  # OpenAQ limit

SESSION = requests.Session()  # reuse HTTP connection


# ----------------------------- SMALL UTILITIES -------------------------------

def _openaq_headers() -> Dict[str, str]:
    """
    Build headers with the OpenAQ API key from env.
    """
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAQ_API_KEY before running.")
    return {"X-API-Key": api_key}


def _iso_utc(dt: datetime) -> str:
    """
    Return UTC ISO8601 string ending with 'Z', e.g. 2025-10-05T12:00:00Z.
    """
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------- GEOCODING + WEATHER -----------------------------

def geocode_city(city: str) -> Tuple[float, float, str]:
    """
    Convert a city name to (lat, lon, timezone) via Open-Meteo geocoding.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}

    r = SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if not data.get("results"):
        raise ValueError(f"City not found: {city}")

    first = data["results"][0]
    latitude = first["latitude"]
    longitude = first["longitude"]
    timezone_name = first["timezone"]
    return latitude, longitude, timezone_name


def fetch_weather(lat: float, lon: float, start: datetime, end: datetime, tz: str) -> pd.DataFrame:
    """
    Download hourly weather (Open-Meteo) between start/end (inclusive).
    Returns columns: ts, temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": tz,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
    }

    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    hourly = js.get("hourly", {})
    if not hourly:
        return pd.DataFrame(columns=["ts"] + WEATHER_VARS)

    df = pd.DataFrame(hourly).rename(columns={"time": "ts"})
    return df


# ------------------------------- OPENAQ v3 -----------------------------------

def find_pm25_location_ids(lat: float, lon: float, radius_m: int = MAX_RADIUS_M, limit: int = 1000) -> List[int]:
    """
    Find OpenAQ location IDs near (lat, lon) with PM2.5 sensors.
    OpenAQ expects "longitude,latitude". If 422, try reversed once.
    """
    url = f"{OPENAQ_BASE}/locations"
    params = {
        "coordinates": f"{lon},{lat}",
        "radius": min(radius_m, MAX_RADIUS_M),
        "parameters_id": 2,  # PM2.5
        "limit": limit,
    }

    r = SESSION.get(url, headers=_openaq_headers(), params=params, timeout=30)

    if r.status_code == 422:  # some examples/docs are inconsistent
        params["coordinates"] = f"{lat},{lon}"
        r = SESSION.get(url, headers=_openaq_headers(), params=params, timeout=30)

    r.raise_for_status()
    js = r.json()

    ids = [loc["id"] for loc in js.get("results", [])]
    print("IDS\n", ids)
    return ids


def get_pm25_sensor_ids_for_location(location_id: int) -> List[int]:
    """
    From a location, list sensors and keep those measuring PM2.5.
    """
    url = f"{OPENAQ_BASE}/locations/{location_id}/sensors"

    r = SESSION.get(url, headers=_openaq_headers(), timeout=30)
    r.raise_for_status()
    js = r.json()

    pm25_ids = []
    results = js.get("results", [])
    print("Result of PM25\n", results)

    for sensor in results:
        param_info = sensor.get("parameter", {})
        is_pm25 = (param_info.get("id") == 2) or (str(param_info.get("name", "")).lower() == "pm25")
        if is_pm25:
            pm25_ids.append(sensor["id"])

    return pm25_ids


def fetch_openaq_pm25_v3(lat: float, lon: float, start_iso: str, end_iso: str) -> pd.DataFrame:
    """
    Fetch hourly PM2.5 from OpenAQ v3:
      1) Find nearby PM2.5 locations
      2) Get PM2.5 sensor IDs
      3) Pull hourly readings per sensor
      4) Average per hour
    Returns: DataFrame ['ts', 'pm25']
    """
    location_ids = find_pm25_location_ids(lat, lon, radius_m=25_000)
    if not location_ids:
        print("No PM2.5 locations found near this city.")
        return pd.DataFrame(columns=["ts", "pm25"])
    print("Found locations:", location_ids)

    sensor_ids = []
    for loc_id in location_ids:
        try:
            sensor_ids.extend(get_pm25_sensor_ids_for_location(loc_id))
            time.sleep(0.1)
        except requests.HTTPError:
            continue

    sensor_ids = list(dict.fromkeys(sensor_ids))  # dedupe
    if not sensor_ids:
        print("No PM2.5 sensors found in those locations.")
        return pd.DataFrame(columns=["ts", "pm25"])
    print("Found sensors:", sensor_ids)

    all_frames = []
    for sid in sensor_ids:
        url = f"{OPENAQ_BASE}/sensors/{sid}/hours"
        params = {"date_from": start_iso, "date_to": end_iso, "limit": 1000}
        try:
            r = requests.get(url, headers=_openaq_headers(), params=params, timeout=30)
            r.raise_for_status()
            data = r.json().get("results", [])
            if not data:
                continue

            rows = []
            for row in data:
                if "datetime" in row and "utc" in row["datetime"] and "value" in row:
                    rows.append({"ts": row["datetime"]["utc"], "pm25": row["value"]})

            if rows:
                df = pd.DataFrame(rows)
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("H")
                all_frames.append(df)
        except requests.HTTPError:
            continue

        time.sleep(0.1)

    if not all_frames:
        print("No hourly readings received from OpenAQ.")
        return pd.DataFrame(columns=["ts", "pm25"])

    combined = pd.concat(all_frames, ignore_index=True)
    hourly_avg = combined.groupby("ts", as_index=False)["pm25"].mean().sort_values("ts")
    print(f"Collected {len(hourly_avg)} hourly records.")
    return hourly_avg


# --------------------------- OPEN-METEO FALLBACK -----------------------------

def fetch_openmeteo_pm25(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fallback: Open-Meteo Air Quality hourly PM2.5 in UTC.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "timezone": "UTC",
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
    }

    r = SESSION.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    hourly = js.get("hourly", {})
    if not hourly or not hourly.get("time"):
        return pd.DataFrame(columns=["ts", "pm25"])

    df = pd.DataFrame({"ts": hourly["time"], "pm25": hourly["pm2_5"]})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    print("hourly\n", df.sort_values("ts"))
    return df.sort_values("ts")


# ------------------------------- MAIN FLOW -----------------------------------

def ingest(city: str = "Karachi", lookback_days: int = 30) -> pd.DataFrame:
    """
    Orchestrate ingestion:
      - compute date window
      - geocode city
      - fetch weather + PM2.5
      - merge on nearest hour (<= 30 min tolerance)
      - save raw snapshots
      - return merged DataFrame
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=lookback_days)

    lat, lon, tz = geocode_city(city)

    weather = fetch_weather(lat, lon, start_utc, now_utc, tz=tz)
    weather["ts"] = pd.to_datetime(weather["ts"])
    weather["ts_utc"] = pd.to_datetime(weather["ts"], utc=True)

    aq_start = _iso_utc(start_utc)
    aq_end = _iso_utc(now_utc)

    pm25 = fetch_openaq_pm25_v3(lat, lon, aq_start, aq_end)
    if pm25.empty:
        print("WARN: OpenAQ returned no PM2.5 rows; using Open-Meteo fallback.")
        pm25 = fetch_openmeteo_pm25(lat, lon, start_utc, now_utc)

    print("pm25 table\n", pm25.head())
    pm25["ts_utc"] = pd.to_datetime(pm25["ts"], utc=True)
    print("pm25 table with utc time\n", pm25.head())

    merged = pd.merge_asof(
        left=pm25.sort_values("ts_utc"),
        right=weather.sort_values("ts_utc"),
        on="ts_utc",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )

    merged = merged.rename(columns={"ts_utc": "ts"})
    merged["ts_local"] = merged["ts"].dt.tz_convert(ZoneInfo("Asia/Karachi"))

    os.makedirs("data/raw", exist_ok=True)

    pm25_out = pm25.drop(columns=["ts"], errors="ignore").rename(columns={"ts_utc": "ts"})
    pm25_out.to_parquet(f"data/raw/pm25_{city}.parquet")
    weather.to_parquet(f"data/raw/weather_{city}.parquet")

    print("Local Time\n", merged["ts_local"])
    return merged


city = os.getenv("CITY", "Karachi")
lookback = int(os.getenv("LOOKBACK_DAYS", "30"))

df = ingest(city=city, lookback_days=lookback)
print(df.head())

if not df.empty:
    df.to_parquet("data/raw/merged_latest.parquet")
    print("Saved merged_latest.parquet with", len(df), "rows")
else:
    print("ERROR: No merged rows; merged_latest not written (check radius/city or fallback).")
