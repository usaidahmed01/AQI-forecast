"""
register_hopsworks.py
=====================
Purpose:
  Upload the 3 trained models from models/latest/ to the Hopsworks Model Registry.

Input expected (from your train.py):
  models/latest/
    - <winner>_tplus24_<version>.joblib
    - <winner>_tplus48_<version>.joblib
    - <winner>_tplus72_<version>.joblib
    - features.json
    - report.json

Env needed:
  HOPSWORKS_PROJECT   (e.g., "AQI-Forecast")
  HOPSWORKS_API_KEY   (your project API key)

How it registers:
  - It registers each horizon as its own "model" in Hopsworks:
      name: aqi_pm25_tplus24   (and 48/72)
    Each run becomes a new version of that model, with files:
      - the specific joblib file
      - features.json
      - report.json
"""

import os
import json
import shutil
from glob import glob
from tempfile import TemporaryDirectory

import hopsworks

MODELS_LATEST_DIR = "models/latest"
HORIZONS = [24, 48, 72]

def _find_model_file_for_horizon(h: int):
    # looks for *_tplus{h}_*.joblib inside models/latest
    pattern = os.path.join(MODELS_LATEST_DIR, f"*tplus{h}_*.joblib")
    hits = glob(pattern)
    return hits[0] if hits else None

def main():
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not project_name or not api_key:
        raise RuntimeError("Set HOPSWORKS_PROJECT and HOPSWORKS_API_KEY environment variables.")

    # connect to Hopsworks
    project = hopsworks.login(project=project_name, api_key_value=api_key)
    mr = project.get_model_registry()

    # read meta for version / feature list
    report_path = os.path.join(MODELS_LATEST_DIR, "report.json")
    feats_path  = os.path.join(MODELS_LATEST_DIR, "features.json")
    if not os.path.exists(report_path) or not os.path.exists(feats_path):
        raise FileNotFoundError("Missing report.json or features.json in models/latest/. Run training first.")

    with open(report_path) as f:
        report = json.load(f)
    with open(feats_path) as f:
        feats = json.load(f)

    version_str = report.get("version", "unknown-version")
    feature_names = feats.get("feature_names", [])

    # register each horizon under a stable name (so you see growing versions)
    for h in HORIZONS:
        model_file = _find_model_file_for_horizon(h)
        if not model_file:
            print(f"[h{h}] No model file found in {MODELS_LATEST_DIR}. Skipping.")
            continue

        # Build a small temp directory with ONLY the files for this horizon.
        with TemporaryDirectory() as tmpdir:
            # copy the horizon model file (joblib)
            dst_model = os.path.join(tmpdir, os.path.basename(model_file))
            shutil.copyfile(model_file, dst_model)
            # copy meta
            shutil.copyfile(report_path, os.path.join(tmpdir, "report.json"))
            shutil.copyfile(feats_path,  os.path.join(tmpdir, "features.json"))

            # Create or get the model entry
            # name like: aqi_pm25_tplus24 (one registry entry per horizon)
            model_name = f"aqi_pm25_tplus{h}"

            # minimal input schema (features list) – helps future users/you
            input_example = {"feature_names": feature_names}

            model = mr.python.create_model(
                name=model_name,
                description=f"PM2.5 forecast +{h}h (version {version_str})",
                input_example=input_example,
                model_schema=None  # keep simple; can add detailed schemas later
            )

            # Save this version (uploads tmpdir content as the model version artifact)
            model.save(model_dir=tmpdir, overwrite=True)

            print(f"[h{h}] Registered {model_name} with version contents from {version_str}")

    print("All done. Check the Hopsworks Model Registry UI for your project.")

if __name__ == "__main__":
    main()
