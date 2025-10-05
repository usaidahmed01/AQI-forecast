import os, joblib, json, numpy as np, pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

TARGET = "pm25_tplus_24"

def load_features(path="data/features/features.parquet"):
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"{path} missing or empty. Run ingest + features first.")
    df = pd.read_parquet(path)
    features = [
        "pm25","pm25_lag1","pm25_lag24","pm25_ma6","pm25_ma24","pm25_chg1",
        "temperature_2m","relative_humidity_2m","wind_speed_10m","surface_pressure",
        "hour","dow","dom","month"
    ]
    need = features + [TARGET]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=need).sort_values("ts")
    X = df[features].values
    y = df[TARGET].values
    return X, y, features

def train_and_eval():
    X, y, features = load_features()
    if len(X) < 200:
        print(f"WARNING: only {len(X)} rows; results may be noisy.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    models = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    }
    best = None
    best_rmse = float("inf")
    report = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rmse = sqrt(mean_squared_error(yte, pred))   # true RMSE
        mae  = mean_absolute_error(yte, pred)
        r2   = r2_score(yte, pred)
        report[name] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        if rmse < best_rmse:
            best, best_rmse = (name, model), rmse

    os.makedirs("models", exist_ok=True)
    joblib.dump(best[1], f"models/{best[0]}_tplus24.joblib")
    with open("models/features.json","w") as f:
        json.dump({"feature_names": features, "target": TARGET}, f, indent=2)
    return report, best[0]

if __name__ == "__main__":
    report, winner = train_and_eval()
    print("Report:", json.dumps(report, indent=2), "Best:", winner)
