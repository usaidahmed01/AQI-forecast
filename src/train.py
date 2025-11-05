"""
Purpose:
  - Train models to forecast PM2.5 at 3 future horizons:
      +24 hours, +48 hours, +72 hours
  - For each horizon, try 2 models (Ridge, RandomForest) and pick the best
  - Save the best model per horizon with versioning and a report

Outputs:
  models/<YYYY-MM-DD_HHMM>/
    - <best>_tplus24.joblib
    - <best>_tplus48.joblib
    - <best>_tplus72.joblib
    - features.json
    - report.json
  models/latest/   (copy of the latest run)

Run:
  python src/train.py
"""

import os
import json
import glob
import shutil
from datetime import datetime
from math import sqrt
from typing import List, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from zoneinfo import ZoneInfo


# ----------------------------- CONFIG ---------------------------------

FEATURES_PATH = "data/features/features.parquet"
MODELS_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# horizons we want to predict
HORIZONS = [24, 48, 72]

# must match features.py
FEATURE_NAMES: List[str] = [
    "pm25",
    "pm25_lag1",
    "pm25_lag24",
    "pm25_ma6",
    "pm25_ma24",
    "pm25_chg1",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "surface_pressure",
    "hour",
    "dow",
    "dom",
    "month",
]

# default target (will be overwritten in loop)
TARGET = "pm25_tplus_24"


# --------------------------- DATA LOADING -----------------------------

def load_features(
    path: str = FEATURES_PATH,
    target_col: str = TARGET,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read the engineered features and return:
      X: features as array
      y: target as array
      features: feature order
    """
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"{path} missing or empty. Run ingest + features first.")

    df = pd.read_parquet(path)

    needed = FEATURE_NAMES + [target_col, "ts"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # keep time order, drop rows with NaNs in needed columns
    df = df.dropna(subset=needed).sort_values("ts")

    X = df[FEATURE_NAMES].values
    y = df[target_col].values

    return X, y, FEATURE_NAMES


# --------------------------- TRAINING CORE ----------------------------

def _train_for_target(
    target_col: str,
) -> Tuple[str, Dict[str, Dict[str, float]], object]:
    """
    Train 2 models for one target and pick the best.

    Returns:
      best_name: name of the best model
      metrics:   metrics for both models
      best_model: fitted model object
    """
    X, y, _ = load_features(FEATURES_PATH, target_col=target_col)

    if len(X) < 200:
        print(f"WARNING: only {len(X)} rows for {target_col}; results may be noisy.")

    # simple time-aware split (no shuffle)
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False,
    )

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    best_name = None
    best_model = None
    best_rmse = float("inf")
    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        rmse = sqrt(mean_squared_error(yte, pred))
        mae = mean_absolute_error(yte, pred)
        r2 = r2_score(yte, pred)

        metrics[name] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        }

        if rmse < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse

    return best_name, metrics, best_model


# --------------------------- MAIN ORCHESTRATION -----------------------

def train_and_eval() -> Tuple[dict, str]:
    """
    Train for 3 horizons, save versioned models, copy to models/latest.

    Returns:
      (report_dict, "ok")
    """
    # version folder based on Asia/Karachi time
    version = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d_%H%M")
    outdir = os.path.join(MODELS_DIR, version)
    os.makedirs(outdir, exist_ok=True)

    overall_report: dict = {
        "version": version,
        "feature_names": FEATURE_NAMES,
        "horizons": {},
    }

    for h in HORIZONS:
        target_col = f"pm25_tplus_{h}"
        print(f"\n=== Training target: {target_col} ===")

        best_name, metrics, best_model = _train_for_target(target_col)

        overall_report["horizons"][f"h{h}"] = {
            "best_model": best_name,
            "metrics": metrics,
        }

        model_path = os.path.join(outdir, f"{best_name}_tplus{h}.joblib")
        joblib.dump(best_model, model_path)
        print(f"Saved: {model_path}")

    # save meta files
    with open(os.path.join(outdir, "features.json"), "w") as f:
        json.dump({"feature_names": FEATURE_NAMES}, f, indent=2)

    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(overall_report, f, indent=2)

    # copy to `models/latest`
    latest_dir = os.path.join(MODELS_DIR, "latest")
    os.makedirs(latest_dir, exist_ok=True)

    for p in glob.glob(os.path.join(outdir, "*")):
        shutil.copy(p, latest_dir)

    print("\nTraining complete. Summary:")
    print(json.dumps(overall_report, indent=2))

    return overall_report, "ok"


# --------------------------- CLI ENTRY --------------------------------

report, winner = train_and_eval()
print("Report:", json.dumps(report, indent=2), "Best:", winner)
