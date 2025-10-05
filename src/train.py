"""
train.py
========
Purpose:
  Train a simple baseline model to predict pm25_tplus_24 (tomorrow's PM2.5).
Models we try:
  - Ridge Regression (linear baseline)
  - Random Forest Regressor (non-linear baseline)
Split:
  Time-aware split: last 20% of rows used as test set (no shuffling),
  so we don't "peek into the future".
Metrics:
  - RMSE: Root Mean Squared Error (lower is better)
  - MAE : Mean Absolute Error     (lower is better)
  - R^2 : Coefficient of Determination (closer to 1 is better)
Saves:
  - models/<best_model_name>_tplus24.joblib  (the fitted model)
  - models/features.json                     (the exact feature order & target, for inference later)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from math import sqrt
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# Where to read training table from, and where to save models
FEATURES_PATH = "data/features/features.parquet"
MODELS_DIR    = "models"
TARGET        = "pm25_tplus_24"

# This list MUST match the columns created in features.py
FEATURE_NAMES: List[str] = [
    "pm25","pm25_lag1","pm25_lag24","pm25_ma6","pm25_ma24","pm25_chg1",
    "temperature_2m","relative_humidity_2m","wind_speed_10m","surface_pressure",
    "hour","dow","dom","month",
]

def load_features(path: str = FEATURES_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read the engineered table and return:
      - X: 2D numpy array of shape (n_samples, n_features)
      - y: 1D numpy array of shape (n_samples,)
      - feature_names: list of strings, used later at inference time
    Also validates that the required columns exist and rows are non-empty.
    """
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"{path} missing or empty. Run ingest + features first.")

    df = pd.read_parquet(path)

    # We need both the inputs (FEATURE_NAMES) and the target column, and a timestamp for sorting
    needed = FEATURE_NAMES + [TARGET, "ts"]
    missing = [c for c in needed if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort chronologically, and drop any rows that still have NaN just in case
    df = df.dropna(subset=needed).sort_values("ts")

    X = df[FEATURE_NAMES].values
    y = df[TARGET].values
    return X, y, FEATURE_NAMES

def train_and_eval() -> Tuple[dict, str]:
    """
    Train both baseline models, evaluate them on the test set, and keep the better one.
    Returns a (metrics_report_dict, best_model_name).
    """
    X, y, features = load_features()

    if len(X) < 200:
        # Not an error, just a friendly warning: small datasets produce unstable metrics
        print(f"WARNING: only {len(X)} training rows; results may be noisy.")

    # Time-aware split: we keep the last 20% of rows as test set (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Two simple models to start
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    }

    best_name = None
    best_model = None
    best_rmse = float("inf")
    metrics_report = {}

    # Fit each model, compute metrics, and keep the winner (lowest RMSE)
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Some sklearn versions don't support squared=False, so we use sqrt(mse) explicitly
        rmse = sqrt(mean_squared_error(y_test, pred))
        mae  = mean_absolute_error(y_test, pred)
        r2   = r2_score(y_test, pred)

        metrics_report[name] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_model = model

    # Save the best model and the feature list used during training
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, f"{best_name}_tplus24.joblib"))
    with open(os.path.join(MODELS_DIR, "features.json"), "w") as f:
        json.dump({"feature_names": features, "target": TARGET}, f, indent=2)

    return metrics_report, best_name

if __name__ == "__main__":
    report, winner = train_and_eval()
    print("Report:", json.dumps(report, indent=2), "Best:", winner)
