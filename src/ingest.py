import os, math, time, requests, pandas as pd
from datetime import datetime, timedelta, timezone

OPENAQ_BASE = "https://api.openaq.org/v3"

def geocode_city(city: str):
    """Free geocoder (Open-Meteo) -> first match lat/lon."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1})
    r.raise_for_status()
    js = r.json()
    if not js.get("results"):
        raise ValueError(f"City not found: {city}")
    res = js["results"][0]
    return res["latitude"], res["longitude"], res["timezone"]

def fetch_weather(lat, lon, start, end, tz="auto"):
    """Open-Meteo hourly weather for [start,end)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure"
        ]),
        "timezone": tz,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "past_days": 0,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    js = r.json()
    hr = js["hourly"]
    df = pd.DataFrame(hr)
    df.rename(columns={"time":"ts"}, inplace=True)
    return df  # columns: ts, temperature_2m, relative_humidity_2m, wind_speed_10m, surface_pressure


def _openaq_headers():
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAQ_API_KEY environment variable.")
    return {"X-API-Key": api_key}

def find_pm25_location_ids(lat: float, lon: float, radius_m: int = 25000, limit: int = 1000):
    """
    Find locations within radius that have PM2.5 sensors.
    Uses /v3/locations with geospatial filter; parameters_id=2 is PM2.5.
    """
    url = f"{OPENAQ_BASE}/locations"
    radius_m = min(radius_m, 25000)
    params = {
        "coordinates": f"{lon},{lat}",   # longitude, latitude per docs
        "radius": radius_m,
        "parameters_id": 2,              # 2 == PM2.5
        "limit": limit
    }
    r = requests.get(url, headers=_openaq_headers(), params=params)
    if r.status_code == 422:
        # Some docs/examples are inconsistent about order.
        # Retry with lat,lon if the first attempt 422s.
        params["coordinates"] = f"{lat},{lon}"
        r = requests.get(url, headers=_openaq_headers(), params=params)
    r.raise_for_status()
    js = r.json()
    return [loc["id"] for loc in js.get("results", [])]

def get_pm25_sensor_ids_for_location(location_id: int):
    """
    List sensors at a location and return those measuring PM2.5.
    /v3/locations/{locations_id}/sensors
    """
    url = f"{OPENAQ_BASE}/locations/{location_id}/sensors"
    r = requests.get(url, headers=_openaq_headers())
    r.raise_for_status()
    js = r.json()
    pm25_ids = []
    for s in js.get("results", []):
    # v3: sensor object -> s["id"]; pollutant -> s["parameter"]["id"] or ["name"]
        p = s.get("parameter", {})
        if p.get("id") == 2 or str(p.get("name", "")).lower() == "pm25":
            pm25_ids.append(s["id"])
    return pm25_ids

def fetch_openaq_pm25_v3(city: str, lat: float, lon: float, start_iso: str, end_iso: str):
    """
    Fetch hourly PM2.5 from OpenAQ v3:
      - find nearby locations
      - get PM2.5 sensors at those locations
      - fetch hourly measurements for each sensor
      - average across sensors per timestamp
    """
    # 1) locations near city
    loc_ids = find_pm25_location_ids(lat, lon, radius_m=25000)
    if not loc_ids:
        return pd.DataFrame(columns=["ts","pm25"])

    # 2) collect PM2.5 sensor IDs
    sensor_ids = []
    for lid in loc_ids:
        try:
            sensor_ids.extend(get_pm25_sensor_ids_for_location(lid))
            time.sleep(0.1)
        except requests.HTTPError:
            continue
    sensor_ids = list(dict.fromkeys(sensor_ids))  # unique

    if not sensor_ids:
        return pd.DataFrame(columns=["ts","pm25"])

    # 3) pull hourly per sensor and stack
    frames = []
    for sid in sensor_ids:
        url = f"{OPENAQ_BASE}/sensors/{sid}/hours"
        params = {
            "date_from": start_iso,
            "date_to": end_iso,
            "limit": 1000
        }
        try:
            r = requests.get(url, headers=_openaq_headers(), params=params)
            r.raise_for_status()
            js = r.json()
            rows = js.get("results", [])
            if not rows:
                continue
            df = pd.DataFrame([{
                "ts": row["datetime"]["utc"],
                "pm25": row["value"]
            } for row in rows if "datetime" in row and "value" in row])
            if not df.empty:
                df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.floor("H")
                frames.append(df)
        except requests.HTTPError:
            continue
        time.sleep(0.1)

    if not frames:
        return pd.DataFrame(columns=["ts","pm25"])

    all_df = pd.concat(frames, ignore_index=True)
    # Average across sensors per hour
    hourly = (all_df.groupby("ts", as_index=False)["pm25"]
                    .mean()
                    .sort_values("ts"))
    return hourly

def fetch_openmeteo_pm25(lat: float, lon: float, start_dt, end_dt):
    """
    Fallback: Open-Meteo Air Quality API (free, no key).
    Returns hourly PM2.5 between dates (UTC).
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5",
        "timezone": "UTC",
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        # Note: Open-Meteo allows limited `past_days` of archived forecasts;
        # for longer histories, we’ll backfill in chunks if needed.
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    js = r.json()
    h = js.get("hourly", {})
    if not h or not h.get("time"):
        return pd.DataFrame(columns=["ts","pm25"])
    df = pd.DataFrame({"ts": h["time"], "pm25": h["pm2_5"]})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts")


def ingest(city="Karachi", lookback_days=30):
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=lookback_days)
    lat, lon, tz = geocode_city(city)
    # Weather: use local timezone label from geocoder for convenience
    weather = fetch_weather(lat, lon, start_utc, now_utc, tz=tz)
    weather["ts"] = pd.to_datetime(weather["ts"])
    # AQI: PM2.5 in UTC ISO
    aq_start = start_utc.isoformat().replace("+00:00","Z")
    aq_end   = now_utc.isoformat().replace("+00:00","Z")
    pm25 = fetch_openaq_pm25_v3(city, lat, lon, aq_start, aq_end)
    if pm25.empty:
        print("WARN: OpenAQ returned no PM2.5 rows; using Open-Meteo fallback.")
        pm25 = fetch_openmeteo_pm25(lat, lon, start_utc, now_utc)


    # Join on timestamp; ensure both in UTC and aligned to hour
    weather["ts_utc"] = pd.to_datetime(weather["ts"], utc=True)
    pm25["ts_utc"]    = pd.to_datetime(pm25["ts"], utc=True)
    df = pd.merge_asof(
        pm25.sort_values("ts_utc"),
        weather.sort_values("ts_utc"),
        on="ts_utc", direction="nearest", tolerance=pd.Timedelta("30min")
    )
    df.rename(columns={"ts_utc":"ts"}, inplace=True)
    # Save raw snapshots
    os.makedirs("data/raw", exist_ok=True)
    pm25.to_parquet(f"data/raw/pm25_{city}.parquet")
    weather.to_parquet(f"data/raw/weather_{city}.parquet")
    return df

if __name__ == "__main__":
    df = ingest(city=os.getenv("CITY","Karachi"), lookback_days=int(os.getenv("LOOKBACK_DAYS", "30")))
    print(df.head())
    
    if not df.empty:
        df.to_parquet("data/raw/merged_latest.parquet")
        print("Saved merged_latest.parquet with", len(df), "rows")
    else:
        print("ERROR: No merged rows; merged_latest not written (check radius/city or fallback).")

