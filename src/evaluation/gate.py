"""Metric-regression gate for CI.

Real gate (default): on real NYC taxi data. CI downloads the public TLC files with
scripts/download_data.py, cleans them with the real cleaning stage, trains the real training code on
a seeded 500k-row sample of Jan-Feb (geo features + L1 objective, the shipped configuration) and validates on all of March. The validation MAE must be no
more than TOLERANCE worse than .github/metric_reference_real.json. Any change to cleaning,
features, training or hyperparameters that makes the model worse fails the PR.

Synthetic gate (--data synthetic): same logic on a tiny generated dataset; only used by unit tests
to exercise the gate mechanism quickly.

After an intentional change, refresh the reference: python -m evaluation.gate --update
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from features import WEATHER_COLUMNS, load_weather, load_zone_categories
from training import train
from validation import run as run_cleaning

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_REAL = ROOT / ".github" / "metric_reference_real.json"
REFERENCE_SYNTHETIC = ROOT / ".github" / "metric_reference_synthetic.json"
TOLERANCE = 0.02  # candidate MAE may be at most 2% worse than the reference
ZONES = list(range(1, 266))
MONTHS = ["2026-01", "2026-02", "2026-03"]
REAL_TRAIN_ROWS = 500_000
# Fixed, single-threaded, deterministic model settings so the gate is stable across machines.
_DETERMINISTIC = {"n_jobs": 1, "deterministic": True, "force_row_wise": True}
GATE_PARAMS = {"n_estimators": 80, "num_leaves": 16, "learning_rate": 0.1, **_DETERMINISTIC}
# The real gate trains the shipped configuration: geo features (use_geo) + L1 objective.
REAL_PARAMS = {"n_estimators": 150, "num_leaves": 63, "learning_rate": 0.1, "objective": "l1", **_DETERMINISTIC}
KEYS = ("mae", "rmse", "p90_abs_error")


def synthetic_trips(seed: int = 0, n_per_month: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for month in MONTHS:
        ts = pd.Timestamp(f"{month}-01") + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, n_per_month), unit="m")
        pu, do = rng.integers(1, 15, n_per_month), rng.integers(1, 15, n_per_month)
        rush = ts.hour.isin([7, 8, 9, 16, 17, 18]).astype(int)
        frames.append(pd.DataFrame({
            "pickup_ts": ts, "PULocationID": pu, "DOLocationID": do, "month": month,
            "duration_min": 6 + 1.5 * pu + 0.8 * do + 4 * rush + rng.normal(0, 2, n_per_month)}))
    return pd.concat(frames, ignore_index=True)


def _rounded(metrics: dict) -> dict[str, float]:
    return {k: round(float(v), 6) for k, v in metrics.items() if k in KEYS}


def gate_metrics() -> dict[str, float]:
    """Synthetic-data metrics (fast; used by unit tests)."""
    trips = synthetic_trips()
    weather = pd.DataFrame({"weather_hour": pd.date_range("2026-01-01", periods=24 * 100, freq="h")})
    for col in WEATHER_COLUMNS:
        weather[col] = 0.0
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # sqlite stays locked on Windows
        uri = f"sqlite:///{(Path(tmp) / 'mlflow.db').as_posix()}"
        return _rounded(train(trips, weather, ZONES, uri, params=GATE_PARAMS)["metrics"])


def real_gate_metrics(raw_dir: Path = ROOT / "data" / "raw") -> dict[str, float]:
    """Real-data metrics: real cleaning + real training on the public TLC files in ``raw_dir``."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        reports = run_cleaning(raw_dir, work / "processed", MONTHS)
        trips = pd.read_parquet(work / "processed" / "trips_clean.parquet")
        result = train(trips, load_weather(raw_dir / "weather_2026_01_07.json"),
                       load_zone_categories(raw_dir / "taxi_zone_lookup.csv"),
                       tracking_uri=f"sqlite:///{(work / 'mlflow.db').as_posix()}", clean_report=reports,
                       max_train_rows=REAL_TRAIN_ROWS, params=REAL_PARAMS, use_geo=True)
    return _rounded(result["metrics"])


def check(candidate: dict, reference: dict, tolerance: float = TOLERANCE) -> tuple[bool, str]:
    limit = reference["mae"] * (1 + tolerance)
    change = (candidate["mae"] - reference["mae"]) / reference["mae"]
    ok = candidate["mae"] <= limit
    verdict = "PASS" if ok else "FAIL"
    return ok, (f"{verdict}: candidate MAE {candidate['mae']:.4f} vs reference {reference['mae']:.4f} "
                f"({change:+.2%}, allowed up to +{tolerance:.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["real", "synthetic"], default="real")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--update", action="store_true", help="overwrite the reference with current metrics")
    args = parser.parse_args()
    real = args.data == "real"
    reference_path = args.reference or (REFERENCE_REAL if real else REFERENCE_SYNTHETIC)
    metrics = real_gate_metrics(args.raw_dir) if real else gate_metrics()
    if args.update:
        reference_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"reference updated ({args.data}): {metrics}")
        sys.exit(0)
    ok, message = check(metrics, json.loads(reference_path.read_text()))
    print(message)
    sys.exit(0 if ok else 1)
