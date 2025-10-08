"""
train.py
========
Purpose:
  - Train models to forecast PM2.5 at 3 future horizons:
      +24 hours, +48 hours, +72 hours
  - For each horizon, try 2 models (Ridge, RandomForest) and pick the best
  - Save the best model per horizon with versioning and a report

What this file outputs:
  models/<YYYY-MM-DD_HHMM>/
    - <best>_tplus24.joblib
    - <best>_tplus48.joblib
    - <best>_tplus72.joblib
    - features.json        (the exact feature order)
    - report.json          (metrics + winners per horizon)
  models/latest/           (a copied, always-up-to-date set of the above)

How to run:
  python src/train.py
"""

import os, json, glob, joblib, numpy as np, pandas as pd
from math import sqrt
from datetime import datetime
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from zoneinfo import ZoneInfo

# --------- CONFIG (keep names same style as your previous code) ---------
FEATURES_PATH = "data/features/features.parquet"
MODELS_DIR    = "models"                 # parent folder for all versions
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

# Horizons we want to predict (+24h, +48h, +72h)
HORIZONS = [24, 48, 72]

# IMPORTANT: keep this list consistent with features.py
FEATURE_NAMES: List[str] = [
    "pm25","pm25_lag1","pm25_lag24","pm25_ma6","pm25_ma24","pm25_chg1",
    "temperature_2m","relative_humidity_2m","wind_speed_10m","surface_pressure",
    "hour","dow","dom","month"
]

# We keep TARGET here for backward compatibility (used as default)
TARGET = "pm25_tplus_24"   # default; we’ll override per horizon in the loop


# --------------------------- DATA LOADING ---------------------------

def load_features(path: str = FEATURES_PATH, target_col: str = TARGET) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Read the engineered features parquet and return:
      X: 2D array of shape (n_samples, n_features)
      y: 1D array of target values for the requested target_col
      features: the feature name order (so inference can build the same order)

    Notes:
      - We validate that both the inputs and the requested target exist
      - We sort by time ('ts') and drop any rows with missing values
    """
    # 1) basic file checks
    if (not os.path.exists(path)) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"{path} missing or empty. Run ingest + features first.")

    df = pd.read_parquet(path)

    # 2) we need FEATURE_NAMES, the chosen target_col, and 'ts' for time-aware sorting
    needed = FEATURE_NAMES + [target_col, "ts"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3) sort chronologically and drop any NaNs (lags/rolling/targets can create NaNs)
    df = df.dropna(subset=needed).sort_values("ts")

    # 4) build X (inputs) and y (labels)
    X = df[FEATURE_NAMES].values
    y = df[target_col].values
    return X, y, FEATURE_NAMES


# --------------------------- TRAINING CORE ---------------------------

def _train_for_target(target_col: str) -> Tuple[str, Dict[str, Dict[str, float]], object]:
    """
    Train 2 models for a single target, evaluate, pick the best.
    Returns:
      best_name: 'ridge' or 'rf'
      metrics:   { 'ridge': {rmse, mae, r2}, 'rf': {...} }
      best_model: the fitted sklearn model object
    """
    X, y, _ = load_features(FEATURES_PATH, target_col=target_col)

    if len(X) < 200:
        print(f"WARNING: only {len(X)} rows for {target_col}; results may be noisy.")

    # Time-aware split: last 20% as hold-out test (no shuffle)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    # Two simple but solid candidates
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "rf": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    best_name, best_model, best_rmse = None, None, float("inf")
    metrics: Dict[str, Dict[str, float]] = {}

    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        # We compute RMSE as sqrt(MSE) to work with any sklearn version
        rmse = sqrt(mean_squared_error(yte, pred))
        mae  = mean_absolute_error(yte, pred)
        r2   = r2_score(yte, pred)

        metrics[name] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        if rmse < best_rmse:
            best_name, best_model, best_rmse = name, model, rmse

    return best_name, metrics, best_model


# --------------------------- MAIN ORCHESTRATION ---------------------------

def train_and_eval() -> Tuple[dict, str]:
    """
    Train for all horizons (24/48/72) in one go, save versioned models, and copy a 'latest' set.

    Returns (report, status_string):
      - report: full JSON-serializable dict with metrics & winners
      - status_string: simple 'ok' for compatibility with your previous usage
    """
    # 1) Create a unique version folder (UTC so it’s consistent in CI):
    #    Example: models/2025-10-08_1845/
    version = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d_%H%M")
    outdir = os.path.join(MODELS_DIR, version)
    os.makedirs(outdir, exist_ok=True)

    # We'll collect everything here for report.json
    overall_report: dict = {
        "version": version,
        "feature_names": FEATURE_NAMES,
        "horizons": {}
    }

    # 2) Train per horizon (+24/+48/+72)
    for h in HORIZONS:
        target_col = f"pm25_tplus_{h}"
        print(f"\n=== Training target: {target_col} ===")

        best_name, metrics, best_model = _train_for_target(target_col)

        # Store metrics + winner in the report
        overall_report["horizons"][f"h{h}"] = {
            "best_model": best_name,
            "metrics": metrics
        }

        # Save the winner model to the versioned folder
        model_path = os.path.join(outdir, f"{best_name}_tplus{h}.joblib")
        joblib.dump(best_model, model_path)
        print(f"Saved: {model_path}")

    # 3) Save features.json (input order) and report.json (metrics + winners)
    with open(os.path.join(outdir, "features.json"), "w") as f:
        json.dump({"feature_names": FEATURE_NAMES}, f, indent=2)
    with open(os.path.join(outdir, "report.json"), "w") as f:
        json.dump(overall_report, f, indent=2)

    # 4) Maintain a convenient "latest" directory (copy everything there)
    latest_dir = os.path.join(MODELS_DIR, "latest")
    os.makedirs(latest_dir, exist_ok=True)
    import shutil
    for p in glob.glob(os.path.join(outdir, "*")):
        shutil.copy(p, latest_dir)

    # 5) Pretty print a summary (like before)
    print("\nTraining complete. Summary:")
    print(json.dumps(overall_report, indent=2))

    # Return a tuple for compatibility with previous pattern (report, winner_string)
    # Here we return 'ok' since we now have multiple winners (one per horizon).
    return overall_report, "ok"


# --------------------------- CLI ENTRY ---------------------------

if __name__ == "__main__":
    report, winner = train_and_eval()
    print("Report:", json.dumps(report, indent=2), "Best:", winner)
