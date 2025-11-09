AQI Forecast — PM2.5 Prediction for Karachi

1. Goal:

    Given the latest hourly PM2.5, weather, and calendar info, predict PM2.5 for the next 24, 48, and 72 hours.

2. Data Pipeline (Ingestion → Features)

    2.1 Ingestion (src/ingest.py)
        Ingestion does 5 things:

            1.Find city coordinates:
                Uses Open-Meteo Geocoding:
                Input: "Karachi"
                Output: latitude, longitude, timezone

            2.Fetch hourly weather:
                Uses Open-Meteo Forecast API for:
                temperature_2m
                relative_humidity_2m
                wind_speed_10m
                surface_pressure

            3.Fetch PM2.5
                Tries OpenAQ v3 first:
                finds nearby locations,
                finds PM2.5 sensors,
                gets hourly values in UTC.
                If that fails, it falls back to Open-Meteo Air Quality.

            4.Merge on hourly timestamp
                It aligns PM2.5 (target) with weather (features) using a 30-min tolerance.

            5.Save to parquet
                data/raw/pm25_Karachi.parquet
                data/raw/weather_Karachi.parquet
                data/raw/merged_latest.parquet

            So after ingestion, we have one table of hourly data with:
            ts (UTC)
            pm25
            weather columns

    2.2 Feature Engineering (src/features.py)
        This script turns the merged hourly table into ML-ready features:

        Time features: hour, dow, dom, month
            because AQI changes with time-of-day and weekdays.

        Lag / rolling features (on pm25):
            pm25_lag1 → 1 hour ago
            pm25_lag24 → 24 hours ago
            pm25_ma6 → 6-hour moving average
            pm25_ma24 → 24-hour moving average
            pm25_chg1 → current – 1 hour ago

            because AQI is highly autocorrelated; yesterday and last hour tell you a lot.

        Future labels:
            pm25_tplus_24
            pm25_tplus_48
            pm25_tplus_72

            These are created with .shift(-h), so the model can learn:

        Drop incomplete rows
            Because lags and future shifts create NaNs at the start and at the end.

    Final output:
        data/features/features.parquet
        This is the file both training and Hopsworks push use.

3. Training Pipeline (src/train.py)

    3.1 What it trains
        For each of the 3 horizons: 24h, 48h, 72h

        it trains 3 models:
        - Ridge(alpha=1.0)
        - RandomForestRegressor(n_estimators=300, n_jobs=-1)
        - GradientBoostingRegressor(random_state=42)

        Then it evaluates them on a time-ordered train/test split and picks the one with the lowest RMSE.

        So every run produces up to 3 winning models:
        best for +24h
        best for +48h
        best for +72h

    3.2 What it saves
        It creates a timestamped folder under models/, like

        Always show details
        models/
            ridge_tplus24.joblib
            rf_tplus48.joblib
            rf_tplus72.joblib
            features.json
            report.json

        features.json -> contains the list of feature columns used in training.

        report.json -> contains metrics for every model for every horizon, plus which one won.

        Then it copies everything to:
        Always show details
        models/latest/

        so that the UI and CI always have a stable place to read from.

4. Streamlit App (app/streamlit_app.py)

    What it does:

        Loads models/latest/ (the 3 joblibs + feature names + report)
        Fetches live forecast weather for Karachi (Open-Meteo)
        Loads recent PM2.5 history from data/raw/pm25_Karachi.parquet
        Rebuilds the same feature as in training:
        rebuild lags on PM2.5
        take current weather
        add calendar features
        line up columns exactly like features.json
        Makes 3 predictions -> +24h, +48h, +72h
        Maps the prediction to an AQI-style label (Good / Moderate / Unhealthy and etc.) and shows colored badges
        Compute a SHAP-style explanation so the user sees which features pushed the forecast up or down.

5. GitHub Actions (Automation)

    You have two workflows:

        5.1 features-hourly.yml
            runs every hour
            does:
                checkout
                install deps
                python src/ingest.py
                python src/features.py
                commit updated features
            goal: keep data fresh

        5.2 train-daily.yml
            runs daily
            does:
                checkout
                install deps
                python src/train.py
                copy winners to models/latest/
                commit them
                detect if Hopsworks secrets exist
                if yes -> switch to Python 3.10, install hsfs[python], and run python src/push_features_hopsworks.py
                upload models as artifact

6. Hopsworks

    Script: src/push_features_hopsworks.py
    Reads data/features/features.parquet
    Normalizes timestamp to UTC
    Connects via hsfs.connection
    Creates/gets feature group aqi_features_hourly version 1
    Inserts the dataframe

    We pinned hsfs[python]==3.9.0rc7 and used Python 3.10 in CI because that was the combo that worked with cloud runner.

7. How to Run Locally

    1. install
    pip install -r requirements.txt

    2. get data
    python src/ingest.py

    3. build features
    python src/features.py

    4. train models
    python src/train.py

    5. run UI
    streamlit run app/streamlit_app.py