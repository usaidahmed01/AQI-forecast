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
"""
register_hopsworks.py

Try to register the 3 horizon models (24/48/72) to Hopsworks Model Registry.

If the Hopsworks Python client in CI cannot create an execution engine
(typical on GitHub Actions with minimal deps), we just print a warning
and exit WITHOUT raising, so the workflow can still succeed.
"""

import os
import json
import shutil
import sys
from glob import glob
from tempfile import TemporaryDirectory
import types

# 1) import hsfs first and patch missing attr so hopsworks import doesn't crash
import hsfs
if not hasattr(hsfs, "hopsworks_udf"):
    hsfs.hopsworks_udf = types.SimpleNamespace(udf=None)

try:
    import hopsworks
except Exception as e:
    print("[register] Could not import hopsworks:", e)
    sys.exit(0)  # don't fail the workflow


MODELS_LATEST_DIR = "models/latest"
HORIZONS = [24, 48, 72]


def _find_model_file_for_horizon(h: int):
    # accept both patterns (*tplus24.joblib and *tplus24_something.joblib)
    pattern = os.path.join(MODELS_LATEST_DIR, f"*tplus{h}*.joblib")
    hits = glob(pattern)
    return hits[0] if hits else None


def main():
    project_name = os.getenv("HOPSWORKS_PROJECT")
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not project_name or not api_key:
        print("[register] HOPSWORKS_PROJECT/API_KEY not set — skipping model registry.")
        sys.exit(0)

    # read meta first so we can at least fail early if missing
    report_path = os.path.join(MODELS_LATEST_DIR, "report.json")
    feats_path = os.path.join(MODELS_LATEST_DIR, "features.json")
    if not os.path.exists(report_path) or not os.path.exists(feats_path):
        print("[register] models/latest/ is missing report.json or features.json — skipping.")
        sys.exit(0)

    with open(report_path) as f:
        report = json.load(f)
    with open(feats_path) as f:
        feats = json.load(f)

    version_str = report.get("version", "unknown-version")
    feature_names = feats.get("feature_names", [])

    # now attempt to log in
    try:
        project = hopsworks.login(project=project_name, api_key_value=api_key)
        mr = project.get_model_registry()
    except Exception as e:
        # this is the error you're seeing: "Couldn't find execution engine ..."
        print("[register] Hopsworks client could not start properly in CI, skipping.")
        print("[register] Details:", repr(e))
        sys.exit(0)

    # if we are here, registry is available — go ahead and register horizons
    for h in HORIZONS:
        model_file = _find_model_file_for_horizon(h)
        if not model_file:
            print(f"[register] (+{h}h) no model file in {MODELS_LATEST_DIR}, skipping.")
            continue

        with TemporaryDirectory() as tmpdir:
            # copy artifacts for this horizon
            dst_model = os.path.join(tmpdir, os.path.basename(model_file))
            shutil.copyfile(model_file, dst_model)
            shutil.copyfile(report_path, os.path.join(tmpdir, "report.json"))
            shutil.copyfile(feats_path, os.path.join(tmpdir, "features.json"))

            model_name = f"aqi_pm25_tplus{h}"
            input_example = {"feature_names": feature_names}

            model = mr.python.create_model(
                name=model_name,
                description=f"PM2.5 forecast +{h}h (version {version_str})",
                input_example=input_example,
                model_schema=None,
            )
            model.save(model_dir=tmpdir, overwrite=True)
            print(f"[register] (+{h}h) registered {model_name}")

    print("[register] Done.")


if __name__ == "__main__":
    main()
