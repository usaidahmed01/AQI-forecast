# """
# Purpose:
#   1) Find a city's latitude/longitude and timezone (using Open-Meteo Geocoding)
#   2) Download hourly weather for a recent date range (Open-Meteo Forecast)
#   3) Download hourly PM2.5 (air quality) using OpenAQ v3
#      - If OpenAQ has no rows near the city, use Open-Meteo Air Quality as a fallback
#   4) Merge weather + PM2.5 on their hourly timestamps
#   5) Save raw snapshots as Parquet files inside data/raw/
# Outputs:
#   - data/raw/pm25_<Karachi>.parquet     (PM2.5 time series)
#   - data/raw/weather_<Karachi>.parquet  (Weather time series)
#   - data/raw/merged_latest.parquet   (Merged PM2.5 + Weather, one row per hour)

# BEFORE RUNNING (in PowerShell on Windows):
#   $env:OPENAQ_API_KEY="YOUR_OPENAQ_KEY"
#   $env:CITY="Karachi"
#   $env:LOOKBACK_DAYS="7"
#   python src/ingest.py
# """

# import os
# import time
# import requests 
# import pandas as pd
# from datetime import datetime, timedelta, timezone
# from typing import Dict, List, Tuple
# from zoneinfo import ZoneInfo
# from dotenv import load_dotenv

# load_dotenv()

# # ------------------------------- CONSTANTS ----------------------------------

# # OpenAQ base URL
# OPENAQ_BASE = "https://api.openaq.org/v3"

# # Which weather variables we want from Open-Meteo
# WEATHER_VARS = [
#     "temperature_2m",
#     "relative_humidity_2m",
#     "wind_speed_10m",
#     "surface_pressure",
# ]

# # OpenAQ limitation: radius around a coordinate query cannot exceed 25 km (25,000 meters)
# MAX_RADIUS_M = 25_000

# # Use a single HTTP session for speed (reuses underlying connection)
# SESSION = requests.Session()

# # ----------------------------- SMALL UTILITIES -------------------------------

# def _openaq_headers() -> Dict[str, str]:
#     """
#     Returns HTTP headers that include your OpenAQ API key.
#     If the environment variable OPENAQ_API_KEY is not set, we stop with a clear error.
#     """
#     api_key = os.getenv("OPENAQ_API_KEY")
#     if not api_key:
#         raise RuntimeError("Please set OPENAQ_API_KEY environment variable before running.")
#     return {"X-API-Key": api_key}

# def _iso_utc(dt: datetime) -> str:
#     """
#     Takes a datetime and returns a UTC ISO8601 string that ends with 'Z'.
#     Example: 2025-10-05T12:00:00Z
#     """
#     # print("Before ISO " ,  dt)
#     # print("After ISO " ,  dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"))
    
#     return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

# # --------------------------- GEOCODING + WEATHER -----------------------------

# def geocode_city(city: str) -> Tuple[float, float, str]:
#     """
#     Step 1) Convert a city name (e.g., 'Karachi') to:
#       - latitude (float)
#       - longitude (float)
#       - timezone string (e.g., 'Asia/Karachi')
#     We use Open-Meteo's FREE geocoding API. No API key needed.
#     """
#     url = "https://geocoding-api.open-meteo.com/v1/search"
#     params = {"name": city, "count": 1}
#     response = SESSION.get(url, params=params, timeout=20)
#     response.raise_for_status()
#     data = response.json()

#     if not data.get("results"):
#         # If the city name is wrong or not found, it's better to stop now with a clear message
#         raise ValueError(f"City not found: {city}")

#     # print(data)
#     first = data["results"][0]
#     latitude = first["latitude"]
#     longitude = first["longitude"]
#     timezone_name = first["timezone"]
#     return latitude, longitude, timezone_name

# def fetch_weather(lat: float, lon: float, start: datetime, end: datetime, tz: str) -> pd.DataFrame:
#     """
#     Step 2) Download hourly weather between start and end dates (inclusive).
#     Uses Open-Meteo forecast API (FREE). We pass the timezone from geocoding to get readable local times.
#     Returns a DataFrame with columns:
#       ts, temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure
#     """
#     url = "https://api.open-meteo.com/v1/forecast"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "hourly": ",".join(WEATHER_VARS),
#         "timezone": tz,                               # e.g., "Asia/Karachi"
#         "start_date": start.strftime("%Y-%m-%d"),     # e.g., "2025-10-01"
#         "end_date": end.strftime("%Y-%m-%d"),
#     }
#     response = SESSION.get(url, params=params, timeout=30)
#     response.raise_for_status()
#     js = response.json()
#     # print(js.get("hourly"))
#     hourly = js.get("hourly", {})
#     if not hourly:
#         # If Open-Meteo didn't return hourly data for some reason, return an empty frame with correct columns
#         return pd.DataFrame(columns=["ts"] + WEATHER_VARS)

#     # Build a table where each row represents one hour
#     df = pd.DataFrame(hourly)
#     # print(df.head())
#     df = df.rename(columns={"time": "ts"})  # rename 'time' column to 'ts'
#     return df

# # ------------------------------- OPENAQ v3 -----------------------------------

# def find_pm25_location_ids(lat: float, lon: float, radius_m: int = MAX_RADIUS_M, limit: int = 1000) -> List[int]:
#     """
#     Step 3A) Ask OpenAQ: "What monitoring locations are within a circle around my city
#     that have PM2.5 sensors?"
#     - OpenAQ expects coordinates as "longitude,latitude" (this order matters!)
#     - We cap radius at 25 km, because OpenAQ rejects larger values for this query type
#     Returns a list of location IDs (integers).
#     """
#     url = f"{OPENAQ_BASE}/locations"
#     params = {
#         "coordinates": f"{lon},{lat}",      # IMPORTANT: lon,lat
#         "radius": min(radius_m, MAX_RADIUS_M),
#         "parameters_id": 2,                 # 2 means PM2.5 in OpenAQ catalog
#         "limit": limit,
#     }
#     response = SESSION.get(url, headers=_openaq_headers(), params=params, timeout=30)

#     # Some docs/examples are confusing; if OpenAQ complains (HTTP 422),
#     # try the reversed order (lat,lon) once.
#     if response.status_code == 422:
#         params["coordinates"] = f"{lat},{lon}"
#         response = SESSION.get(url, headers=_openaq_headers(), params=params, timeout=30)

#     response.raise_for_status()
#     js = response.json()
#     ids = [loc["id"] for loc in js.get("results", [])]
    
#     print("IDS \n" , ids)
#     return ids

# def get_pm25_sensor_ids_for_location(location_id: int) -> List[int]:
#     """
#     Step 3B) For a given location ID, list its sensors and keep only those measuring PM2.5.
#     We check sensor['parameter']['id'] == 2 (PM2.5), or name == 'pm25'.
#     Returns a list of sensor IDs (integers).
#     """
#     url = f"{OPENAQ_BASE}/locations/{location_id}/sensors"
#     response = SESSION.get(url, headers=_openaq_headers(), timeout=30)
#     response.raise_for_status()
#     js = response.json()

#     pm25_ids = []
#     results = js.get("results", [])
#     print("Result of PM25 \n" , results)
#     for sensor in results:
#         param_info = sensor.get("parameter", {})
#         is_pm25 = (param_info.get("id") == 2) or (str(param_info.get("name", "")).lower() == "pm25")
#         if is_pm25:
#             pm25_ids.append(sensor["id"])
#     return pm25_ids


# def fetch_openaq_pm25_v3(lat: float, lon: float, start_iso: str, end_iso: str) -> pd.DataFrame:
#     """
#     Fetch hourly PM2.5 data from OpenAQ v3 for a city (based on lat/lon).
#     Steps:
#       1. Find nearby monitoring stations with PM2.5 sensors
#       2. Get each sensor's ID
#       3. Download hourly readings for each sensor in the date range
#       4. Merge & average them by hour
#     Returns:
#       DataFrame with columns ['ts', 'pm25']
#     """
#     # Step 1: find nearby locations that measure PM2.5
#     location_ids = find_pm25_location_ids(lat, lon, radius_m=25000)
#     if not location_ids:
#         print("No PM2.5 locations found near this city.")
#         return pd.DataFrame(columns=["ts", "pm25"])
#     print("Found locations:", location_ids)

#     # Step 2: collect all PM2.5 sensor IDs from those locations
#     sensor_ids = []
#     for loc_id in location_ids:
#         try:
#             sensor_ids.extend(get_pm25_sensor_ids_for_location(loc_id))
#             time.sleep(0.1)
#         except requests.HTTPError:
#             continue
#     sensor_ids = list(dict.fromkeys(sensor_ids))  # remove duplicates
#     if not sensor_ids:
#         print("No PM2.5 sensors found in those locations.")
#         return pd.DataFrame(columns=["ts", "pm25"])
#     print("Found sensors:", sensor_ids)

#     # Step 3: fetch hourly readings for each sensor
#     all_frames = []
#     for sid in sensor_ids:
#         url = f"{OPENAQ_BASE}/sensors/{sid}/hours"
#         params = {"date_from": start_iso, "date_to": end_iso, "limit": 1000}
#         try:
#             r = requests.get(url, headers=_openaq_headers(), params=params, timeout=30)
#             r.raise_for_status()
#             data = r.json().get("results", [])
#             if not data:
#                 continue
#             # Extract timestamp and value safely
#             rows = [
#                 {"ts": row["datetime"]["utc"], "pm25": row["value"]}
#                 for row in data
#                 if "datetime" in row and "utc" in row["datetime"] and "value" in row
#             ]
#             df = pd.DataFrame(rows)
#             df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("H")
#             all_frames.append(df)
#         except requests.HTTPError:
#             continue
#         time.sleep(0.1)

#     # Step 4: merge all readings and average by hour
#     if not all_frames:
#         print("No hourly readings received from OpenAQ.")
#         return pd.DataFrame(columns=["ts", "pm25"])

#     combined = pd.concat(all_frames, ignore_index=True)
#     hourly_avg = combined.groupby("ts", as_index=False)["pm25"].mean().sort_values("ts")
#     print(f"Collected {len(hourly_avg)} hourly records.")
#     return hourly_avg


# # --------------------------- OPEN-METEO FALLBACK -----------------------------

# def fetch_openmeteo_pm25(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
#     """
#     If OpenAQ returns no rows (maybe no active sensors nearby), we use Open-Meteo Air Quality.
#     It’s FREE and returns hourly PM2.5 for the date range in UTC.
#     """
#     url = "https://air-quality-api.open-meteo.com/v1/air-quality"
#     params = {
#         "latitude": lat,
#         "longitude": lon,
#         "hourly": "pm2_5",
#         "timezone": "UTC",  # we want UTC to align easily
#         "start_date": start_dt.strftime("%Y-%m-%d"),
#         "end_date": end_dt.strftime("%Y-%m-%d"),
#     }
#     response = SESSION.get(url, params=params, timeout=30)
#     response.raise_for_status()
#     js = response.json()
#     hourly = js.get("hourly", {})
#     if not hourly or not hourly.get("time"):
#         return pd.DataFrame(columns=["ts", "pm25"])

#     df = pd.DataFrame({"ts": hourly["time"], "pm25": hourly["pm2_5"]})
#     df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
#     print("hourly \n" , df.sort_values('ts'))
    
#     return df.sort_values("ts")

# # ------------------------------- MAIN FLOW -----------------------------------

# def ingest(city: str = "Karachi", lookback_days: int = 30) -> pd.DataFrame:
#     """
#     Orchestrates the ingestion:
#       - figure out the date window (last N days to now)
#       - geocode the city
#       - download weather + pm25
#       - merge on nearest hour (<= 30 minutes tolerance)
#       - save raw snapshots (pm25 + weather)
#       - return merged dataframe
#     """
#     now_utc = datetime.now(timezone.utc)
#     start_utc = now_utc - timedelta(days=lookback_days)

#     # 1) city -> (lat, lon, timezone)
#     lat, lon, tz = geocode_city(city)

#     # 2) weather table (local tz labels, plus UTC copy)
#     weather = fetch_weather(lat, lon, start_utc, now_utc, tz=tz)
#     # print("Before \n" , weather.head())
#     weather["ts"] = pd.to_datetime(weather["ts"])                # parse whatever tz label Open-Meteo used
#     # print("After \n" , weather.head())
#     weather["ts_utc"] = pd.to_datetime(weather["ts"], utc=True)  # canonical UTC
#     # print("Acc to UTC \n" , weather.head())

#     # 3) pm25 table (prefer OpenAQ, else fallback to Open-Meteo air quality)
#     aq_start = _iso_utc(start_utc)
#     aq_end   = _iso_utc(now_utc)
#     pm25 = fetch_openaq_pm25_v3(lat, lon, aq_start, aq_end)
#     if pm25.empty:
#         print("WARN: OpenAQ returned no PM2.5 rows; using Open-Meteo fallback.")
#         pm25 = fetch_openmeteo_pm25(lat, lon, start_utc, now_utc)
#     print("pm25 table \n" , pm25.head())
#     pm25["ts_utc"] = pd.to_datetime(pm25["ts"], utc=True)
#     print("pm25 table with utc time \n" , pm25.head())
    

#     # 4) merge the two tables by time — we merge "as of" the nearest timestamp,
#     #    but only if the gap is <= 30 minutes to avoid wrong matches
#     merged = pd.merge_asof(
#         left=pm25.sort_values("ts_utc"),
#         right=weather.sort_values("ts_utc"),
#         on="ts_utc",
#         direction="nearest",
#         tolerance=pd.Timedelta("30min"),
#     )
#     merged = merged.rename(columns={"ts_utc": "ts"})  # final time column is 'ts' in UTC
#     merged['ts_local'] = merged['ts'].dt.tz_convert(ZoneInfo("Asia/Karachi"))

#     # 5) save raw snapshots (for reproducibility and debugging)
#     os.makedirs("data/raw", exist_ok=True)
#     # Save PM2.5 as (ts, pm25): keep the UTC time column name as 'ts' for consistency
#     pm25_out = pm25.drop(columns=["ts"], errors="ignore").rename(columns={"ts_utc": "ts"})
#     pm25_out.to_parquet(f"data/raw/pm25_{city}.parquet")
#     weather.to_parquet(f"data/raw/weather_{city}.parquet")
    
#     print("Local Time \n", merged['ts_local'])

#     return merged

# if __name__ == "__main__":
#     # Allow environment variables to override defaults
#     city = os.getenv("CITY", "Karachi")
#     lookback = int(os.getenv("LOOKBACK_DAYS", "30"))

#     df = ingest(city=city, lookback_days=lookback)
#     print(df.head())

#     # Only write merged_latest if we actually have rows
#     if not df.empty:
#         df.to_parquet("data/raw/merged_latest.parquet")
#         print("Saved merged_latest.parquet with", len(df), "rows")
#     else:
#         print("ERROR: No merged rows; merged_latest not written (check radius/city or fallback).")


























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


if __name__ == "__main__":
    city = os.getenv("CITY", "Karachi")
    lookback = int(os.getenv("LOOKBACK_DAYS", "30"))

    df = ingest(city=city, lookback_days=lookback)
    print(df.head())

    if not df.empty:
        df.to_parquet("data/raw/merged_latest.parquet")
        print("Saved merged_latest.parquet with", len(df), "rows")
    else:
        print("ERROR: No merged rows; merged_latest not written (check radius/city or fallback).")
