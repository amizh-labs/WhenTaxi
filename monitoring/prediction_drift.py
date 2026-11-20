"""Drift on REAL requests: compare the logged /predict window with a training-reference sample (Jan-Feb 2026).

    python -m monitoring.prediction_drift --log <path> [--out monitoring/reports]

What can trigger a retrain (``dataset_drift``): only the TRIGGER columns, hour, day of week (``dow``) and the geo
distance between the zone centroids. They describe WHEN and WHERE people ride, which is what the model must keep
matching. The four weather columns are reported as CONTEXT (``context_drift``) but never set ``dataset_drift``:
weather changes with the season (a November log will differ from January-February) without the model being wrong,
and it is already an explicit model input. Raw zone ids are not compared at all: with ~260 categories and a few
hundred requests almost every id is rare, so a test on them would flag noise; distance carries the "where" signal.

Per-column test: Kolmogorov-Smirnov at p < KS_ALPHA (0.003). Unlike Evidently's default fixed-distance rule, a
p-value test accounts for the sample size, so a small window is not flagged by sampling noise alone. With three
trigger columns the chance of a false ``dataset_drift`` is about 1% (tests/test_prediction_log.py measures it).
dataset_drift is true when at least one trigger column drifts.

Report (docs/contracts.md sections 5 and 8): ``month``, ``dataset_drift``, ``drifted_columns`` (trigger columns
only), ``share_drifted``, ``context_drift`` (weather), ``data_quality`` (share of requests with missing weather and
an alert flag; never sets dataset_drift), plus the keys flows/retrain.py already reads (``share_drifted_features``,
``drifted_features``), ``window_start``, ``window_end``, ``n_rows``, ``status``.
Below MIN_ROWS logged requests (or with no log) the status is "insufficient data" and dataset_drift is false.
The check assumes representative traffic: demo drift runs use monitoring/simulator.py.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from features import WEATHER_COLUMNS, build_features, geo_features, load_weather
from monitoring.drift import compute_drift
from serving.prediction_log import read_log

ROOT = Path(__file__).resolve().parents[1]
MIN_ROWS = 200
REFERENCE_MONTHS = ["2026-01", "2026-02"]
REFERENCE_SAMPLE = 20_000
TRIGGER_COLUMNS = ["hour", "dow", "distance_km"]
CONTEXT_COLUMNS = list(WEATHER_COLUMNS)
LOG_DRIFT_COLUMNS = [*TRIGGER_COLUMNS, *CONTEXT_COLUMNS]
KS_ALPHA = 0.003  # Bonferroni-style: 3 trigger columns -> about 1% chance of a false dataset_drift
MISSING_WEATHER_ALERT = 0.20
INSUFFICIENT = "insufficient data"


def reference_features(trips: pd.DataFrame, weather: pd.DataFrame, months=None, n: int = REFERENCE_SAMPLE,
                       seed: int = 0) -> pd.DataFrame:
    """Training reference: a seeded sample of the training months, with exactly the drift columns."""
    rows = trips[trips["month"].isin(months or REFERENCE_MONTHS)]
    if len(rows) > n:
        rows = rows.sample(n, random_state=seed)
    return build_features(rows, weather, None, geo=True)[LOG_DRIFT_COLUMNS].reset_index(drop=True)


def log_features(log: pd.DataFrame) -> pd.DataFrame:
    """The drift columns for logged requests (hour/dow from pickup_ts, distance from the zone ids)."""
    pickup = pd.to_datetime(log["pickup_ts"])
    out = pd.DataFrame({"hour": pickup.dt.hour, "dow": pickup.dt.day_of_week})
    for col in WEATHER_COLUMNS:
        out[col] = pd.to_numeric(log[col], errors="coerce")
    out["distance_km"] = geo_features(log["pickup_zone"].astype(int), log["dropoff_zone"].astype(int))["distance_km"]
    return out[LOG_DRIFT_COLUMNS].astype(float).reset_index(drop=True)


def data_quality(current: pd.DataFrame) -> dict:
    """Share of logged requests with ANY weather value missing. Alerts above 20%; never a drift trigger."""
    share = float(current[CONTEXT_COLUMNS].isna().any(axis=1).mean()) if len(current) else 0.0
    return {"share_missing_weather": round(share, 4), "alert": share > MISSING_WEATHER_ALERT}


def _report(status: str, trigger: list[str], context: list[str], compared_context: list[str], start: str, end: str,
            n_rows: int, quality: dict) -> dict:
    share = round(len(trigger) / len(TRIGGER_COLUMNS), 4)
    return {
        "month": start[:10] if start else "n/a",
        "dataset_drift": len(trigger) >= 1,  # a retrain trigger needs a trigger column; weather never counts
        "drifted_columns": trigger, "share_drifted": share,
        "context_drift": {"columns": compared_context, "drifted_columns": context,
                          "share_drifted": round(len(context) / max(len(compared_context), 1), 4)},
        "data_quality": quality,
        "share_drifted_features": share, "drifted_features": trigger,  # keys already read by flows/retrain.py
        "window_start": start, "window_end": end, "n_rows": n_rows, "min_rows": MIN_ROWS, "status": status,
    }


def check_log_drift(reference: pd.DataFrame, log: pd.DataFrame, min_rows: int = MIN_ROWS,
                    html_path: Path | None = None) -> dict:
    n = len(log)
    start = str(log["ts"].min()) if n else ""
    end = str(log["ts"].max()) if n else ""
    current = log_features(log) if n else pd.DataFrame(columns=LOG_DRIFT_COLUMNS)
    quality = data_quality(current)
    if n < min_rows:
        return _report(INSUFFICIENT, [], [], [], start, end, n, quality) | {"dataset_drift": False}
    # Weather columns with no values at all cannot be compared; that shows up in data_quality instead.
    compared = [c for c in LOG_DRIFT_COLUMNS if not current[c].isna().all()]
    result = compute_drift(reference, current, start, end, columns=compared, html_path=html_path,
                           method="ks", threshold=KS_ALPHA)
    drifted = set(result["drifted_features"])
    return _report("ok", sorted(drifted & set(TRIGGER_COLUMNS)), sorted(drifted & set(CONTEXT_COLUMNS)),
                   [c for c in compared if c in CONTEXT_COLUMNS], start, end, n, quality)


def run(log_path: Path, trips_path: Path, weather_path: Path, out: Path, min_rows: int = MIN_ROWS) -> dict:
    """Read the log, compare it with the reference, write drift_report.json (+ HTML) into ``out``."""
    log = read_log(log_path)
    if len(log) < min_rows:  # nothing to compare: skip loading the training data
        result = check_log_drift(pd.DataFrame(columns=LOG_DRIFT_COLUMNS), log, min_rows)
    else:
        reference = reference_features(pd.read_parquet(trips_path), load_weather(weather_path))
        result = check_log_drift(reference, log, min_rows, html_path=out / "drift_report.html")
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", type=Path,
                   default=Path(os.environ.get("ETA_PREDICTION_LOG", ROOT / "data" / "prediction_log.sqlite")))
    p.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    p.add_argument("--weather", type=Path, default=ROOT / "data" / "raw" / "weather_2026_01_07.json")
    p.add_argument("--min-rows", type=int, default=MIN_ROWS)
    p.add_argument("--out", type=Path, default=ROOT / "monitoring" / "reports")
    args = p.parse_args(argv)
    print(json.dumps(run(args.log, args.trips, args.weather, args.out, args.min_rows), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
