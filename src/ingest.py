"""
ingest.py
=========
Purpose:
  1) Find a city's latitude/longitude and timezone (using Open-Meteo Geocoding)
  2) Download hourly weather for a recent date range (Open-Meteo Forecast)
  3) Download hourly PM2.5 (air quality) using OpenAQ v3
     - If OpenAQ has no rows near the city, use Open-Meteo Air Quality as a fallback
  4) Merge weather + PM2.5 on their hourly timestamps
  5) Save raw snapshots as Parquet files inside data/raw/
Outputs:
  - data/raw/pm25_<city>.parquet     (PM2.5 time series)
  - data/raw/weather_<city>.parquet  (Weather time series)
  - data/raw/merged_latest.parquet   (Merged PM2.5 + Weather, one row per hour)

BEFORE RUNNING (in PowerShell on Windows):
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

# ------------------------------- CONSTANTS ----------------------------------

# OpenAQ base URL (version 3)
OPENAQ_BASE = "https://api.openaq.org/v3"

# Which weather variables we want from Open-Meteo
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
]

# OpenAQ limitation: radius around a coordinate query cannot exceed 25 km (25,000 meters)
MAX_RADIUS_M = 25_000

# Use a single HTTP session for speed (reuses underlying connection)
SESSION = requests.Session()

# ----------------------------- SMALL UTILITIES -------------------------------

def _openaq_headers() -> Dict[str, str]:
    """
    Returns HTTP headers that include your OpenAQ API key.
    If the environment variable OPENAQ_API_KEY is not set, we stop with a clear error.
    """
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAQ_API_KEY environment variable before running.")
    return {"X-API-Key": api_key}

def _iso_utc(dt: datetime) -> str:
    """
    Takes a datetime and returns a UTC ISO8601 string that ends with 'Z'.
    Example: 2025-10-05T12:00:00Z
    """
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

# --------------------------- GEOCODING + WEATHER -----------------------------

def geocode_city(city: str) -> Tuple[float, float, str]:
    """
    Step 1) Convert a city name (e.g., 'Karachi') to:
      - latitude (float)
      - longitude (float)
      - timezone string (e.g., 'Asia/Karachi')
    We use Open-Meteo's FREE geocoding API. No API key needed.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}
    response = SESSION.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    if not data.get("results"):
        # If the city name is wrong or not found, it's better to stop now with a clear message
        raise ValueError(f"City not found: {city}")

    first = data["results"][0]
    latitude = first["latitude"]
    longitude = first["longitude"]
    timezone_name = first["timezone"]
    return latitude, longitude, timezone_name

def fetch_weather(lat: float, lon: float, start: datetime, end: datetime, tz: str) -> pd.DataFrame:
    """
    Step 2) Download hourly weather between start and end dates (inclusive).
    Uses Open-Meteo forecast API (FREE). We pass the timezone from geocoding to get readable local times.
    Returns a DataFrame with columns:
      ts, temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": tz,                               # e.g., "Asia/Karachi"
        "start_date": start.strftime("%Y-%m-%d"),     # e.g., "2025-10-01"
        "end_date": end.strftime("%Y-%m-%d"),
    }
    response = SESSION.get(url, params=params, timeout=30)
    response.raise_for_status()
    js = response.json()
    hourly = js.get("hourly", {})
    if not hourly:
        # If Open-Meteo didn't return hourly data for some reason, return an empty frame with correct columns
        return pd.DataFrame(columns=["ts"] + WEATHER_VARS)

    # Build a table where each row represents one hour
    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "ts"})  # rename 'time' column to 'ts'
    return df

# ------------------------------- OPENAQ v3 -----------------------------------

def find_pm25_location_ids(lat: float, lon: float, radius_m: int = MAX_RADIUS_M, limit: int = 1000) -> List[int]:
    """
    Step 3A) Ask OpenAQ: "What monitoring locations are within a circle around my city
    that have PM2.5 sensors?"
    - OpenAQ expects coordinates as "longitude,latitude" (this order matters!)
    - We cap radius at 25 km, because OpenAQ rejects larger values for this query type
    Returns a list of location IDs (integers).
    """
    url = f"{OPENAQ_BASE}/locations"
    params = {
        "coordinates": f"{lon},{lat}",      # IMPORTANT: lon,lat
        "radius": min(radius_m, MAX_RADIUS_M),
        "parameters_id": 2,                 # 2 means PM2.5 in OpenAQ catalog
        "limit": limit,
    }
    response = SESSION.get(url, headers=_openaq_headers(), params=params, timeout=30)

    # Some docs/examples are confusing; if OpenAQ complains (HTTP 422),
    # try the reversed order (lat,lon) once.
    if response.status_code == 422:
        params["coordinates"] = f"{lat},{lon}"
        response = SESSION.get(url, headers=_openaq_headers(), params=params, timeout=30)

    response.raise_for_status()
    js = response.json()
    ids = [loc["id"] for loc in js.get("results", [])]
    return ids

def get_pm25_sensor_ids_for_location(location_id: int) -> List[int]:
    """
    Step 3B) For a given location ID, list its sensors and keep only those measuring PM2.5.
    We check sensor['parameter']['id'] == 2 (PM2.5), or name == 'pm25'.
    Returns a list of sensor IDs (integers).
    """
    url = f"{OPENAQ_BASE}/locations/{location_id}/sensors"
    response = SESSION.get(url, headers=_openaq_headers(), timeout=30)
    response.raise_for_status()
    js = response.json()

    pm25_ids = []
    results = js.get("results", [])
    for sensor in results:
        param_info = sensor.get("parameter", {})
        is_pm25 = (param_info.get("id") == 2) or (str(param_info.get("name", "")).lower() == "pm25")
        if is_pm25:
            pm25_ids.append(sensor["id"])
    return pm25_ids

def fetch_openaq_pm25_v3(lat: float, lon: float, start_iso: str, end_iso: str) -> pd.DataFrame:
    """
    Step 3C) Pull hourly PM2.5 values from OpenAQ v3:
      1) Find nearby locations (within radius) that have PM2.5 sensors
      2) For each location, get the PM2.5 sensor IDs
      3) For each sensor, request its hourly values in the time window
      4) Combine everything and average across sensors for each hour
    Returns a DataFrame with columns: ts, pm25
    """
    # 1) which locations near the city have PM2.5?
    location_ids = find_pm25_location_ids(lat, lon, radius_m=MAX_RADIUS_M)
    if not location_ids:
        return pd.DataFrame(columns=["ts", "pm25"])

    # 2) collect PM2.5 sensor IDs (unique list)
    sensor_ids: List[int] = []
    for loc_id in location_ids:
        try:
            sensor_ids.extend(get_pm25_sensor_ids_for_location(loc_id))
            time.sleep(0.1)  # small pause so we don't hammer the API
        except requests.HTTPError:
            # if one location fails, skip it and continue
            continue
    sensor_ids = list(dict.fromkeys(sensor_ids))  # remove duplicates in order
    if not sensor_ids:
        return pd.DataFrame(columns=["ts", "pm25"])

    # 3) read hourly values for each sensor
    frames: List[pd.DataFrame] = []
    for sid in sensor_ids:
        url = f"{OPENAQ_BASE}/sensors/{sid}/hours"
        params = {"date_from": start_iso, "date_to": end_iso, "limit": 1000}
        try:
            response = SESSION.get(url, headers=_openaq_headers(), params=params, timeout=30)
            response.raise_for_status()
            rows = response.json().get("results", [])
            if len(rows) == 0:
                continue

            # Build a table for this sensor
            # We also guard against missing keys to avoid KeyError
            table_rows = []
            for row in rows:
                utc_str = None
                if "datetime" in row and isinstance(row["datetime"], dict):
                    utc_str = row["datetime"].get("utc")
                value = row.get("value")
                if utc_str is not None and value is not None:
                    table_rows.append({"ts": utc_str, "pm25": value})

            if len(table_rows) > 0:
                df = pd.DataFrame(table_rows)
                # convert to UTC pandas timestamps and align exactly to the hour
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("H")
                frames.append(df)
        except requests.HTTPError:
            # if a sensor fails, ignore it and continue with others
            continue

        time.sleep(0.1)  # be gentle

    if len(frames) == 0:
        return pd.DataFrame(columns=["ts", "pm25"])

    # 4) stack all sensors and average per hour
    all_rows = pd.concat(frames, ignore_index=True)
    hourly = (
        all_rows.groupby("ts", as_index=False)["pm25"]
        .mean()
        .sort_values("ts")
    )
    return hourly

# --------------------------- OPEN-METEO FALLBACK -----------------------------

def fetch_openmeteo_pm25(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    If OpenAQ returns no rows (maybe no active sensors nearby), we use Open-Meteo Air Quality.
    It’s FREE and returns hourly PM2.5 for the date range in UTC.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "timezone": "UTC",  # we want UTC to align easily
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
    }
    response = SESSION.get(url, params=params, timeout=30)
    response.raise_for_status()
    js = response.json()
    hourly = js.get("hourly", {})
    if not hourly or not hourly.get("time"):
        return pd.DataFrame(columns=["ts", "pm25"])

    df = pd.DataFrame({"ts": hourly["time"], "pm25": hourly["pm2_5"]})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")

# ------------------------------- MAIN FLOW -----------------------------------

def ingest(city: str = "Karachi", lookback_days: int = 30) -> pd.DataFrame:
    """
    Orchestrates the ingestion:
      - figure out the date window (last N days to now)
      - geocode the city
      - download weather + pm25
      - merge on nearest hour (<= 30 minutes tolerance)
      - save raw snapshots (pm25 + weather)
      - return merged dataframe
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=lookback_days)

    # 1) city -> (lat, lon, timezone)
    lat, lon, tz = geocode_city(city)

    # 2) weather table (local tz labels, plus UTC copy)
    weather = fetch_weather(lat, lon, start_utc, now_utc, tz=tz)
    weather["ts"] = pd.to_datetime(weather["ts"])              # parse whatever tz label Open-Meteo used
    weather["ts_utc"] = pd.to_datetime(weather["ts"], utc=True)  # canonical UTC

    # 3) pm25 table (prefer OpenAQ, else fallback to Open-Meteo air quality)
    aq_start = _iso_utc(start_utc)
    aq_end   = _iso_utc(now_utc)
    pm25 = fetch_openaq_pm25_v3(lat, lon, aq_start, aq_end)
    if pm25.empty:
        print("WARN: OpenAQ returned no PM2.5 rows; using Open-Meteo fallback.")
        pm25 = fetch_openmeteo_pm25(lat, lon, start_utc, now_utc)
    pm25["ts_utc"] = pd.to_datetime(pm25["ts"], utc=True)

    # 4) merge the two tables by time — we merge "as of" the nearest timestamp,
    #    but only if the gap is <= 30 minutes to avoid wrong matches
    merged = pd.merge_asof(
        left=pm25.sort_values("ts_utc"),
        right=weather.sort_values("ts_utc"),
        on="ts_utc",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    merged = merged.rename(columns={"ts_utc": "ts"})  # final time column is 'ts' in UTC

    # 5) save raw snapshots (for reproducibility and debugging)
    os.makedirs("data/raw", exist_ok=True)
    # Save PM2.5 as (ts, pm25): keep the UTC time column name as 'ts' for consistency
    pm25_out = pm25.drop(columns=["ts"], errors="ignore").rename(columns={"ts_utc": "ts"})
    pm25_out.to_parquet(f"data/raw/pm25_{city}.parquet")
    weather.to_parquet(f"data/raw/weather_{city}.parquet")

    return merged

if __name__ == "__main__":
    # Allow environment variables to override defaults
    city = os.getenv("CITY", "Karachi")
    lookback = int(os.getenv("LOOKBACK_DAYS", "30"))

    df = ingest(city=city, lookback_days=lookback)
    print(df.head())

    # Only write merged_latest if we actually have rows
    if not df.empty:
        df.to_parquet("data/raw/merged_latest.parquet")
        print("Saved merged_latest.parquet with", len(df), "rows")
    else:
        print("ERROR: No merged rows; merged_latest not written (check radius/city or fallback).")
