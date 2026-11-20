"""Evidently drift detection. Writes drift_report.json (docs/contracts.md section 5).

Compares a reference window (e.g. the validation month) with a current window of feature
rows (+ duration_min when known). The dataset-level decision is ours, not Evidently's
default "50% of columns" rule: one storm only moves a few columns, so we flag drift when
>= SHARE_THRESHOLD of the compared columns drift.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from features import WEATHER_COLUMNS, build_features, load_weather, load_zone_categories

ROOT = Path(__file__).resolve().parents[1]
SHARE_THRESHOLD = 0.2
DRIFT_COLUMNS = ["hour", "dow", "is_weekend", *WEATHER_COLUMNS, "duration_min"]
_COLUMN_RE = re.compile(r"column=(?P<col>[^,]+),.*threshold=(?P<thr>[\d.]+)")


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, window_start: str, window_end: str,
                  columns: list[str] | None = None, share_threshold: float = SHARE_THRESHOLD,
                  html_path: Path | None = None, method: str | None = None, threshold: float | None = None) -> dict:
    """method/threshold: optional Evidently stat test for every column (e.g. "ks", 0.005). Default: Evidently's."""
    columns = [c for c in (columns or DRIFT_COLUMNS) if c in reference.columns and c in current.columns]
    preset = (DataDriftPreset(columns=columns, method=method, threshold=threshold) if method
              else DataDriftPreset(columns=columns))
    snapshot = Report([preset]).run(
        Dataset.from_pandas(current[columns].astype(float)), Dataset.from_pandas(reference[columns].astype(float)))
    drifted = []
    for metric in snapshot.dict()["metrics"]:
        match = _COLUMN_RE.search(metric["metric_name"])
        if not match:
            continue
        # p-value tests (ks, ...) drift when the value is BELOW the threshold; distance tests when it is above.
        is_p_value = "method=ks" in metric["metric_name"].replace(" ", "")
        value, thr = float(metric["value"]), float(match["thr"])
        if (value < thr) if is_p_value else (value > thr):
            drifted.append(match["col"])
    share = len(drifted) / len(columns)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    return {
        "dataset_drift": share >= share_threshold,
        "share_drifted_features": round(share, 4),
        "drifted_features": sorted(drifted),
        "window_start": window_start,
        "window_end": window_end,
    }


def month_frame(trips: pd.DataFrame, month: str, weather, zones) -> pd.DataFrame:
    return build_features(trips[trips["month"] == month], weather, zones)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="2026-03", help="month in trips_clean.parquet")
    parser.add_argument("--current", default="2026-04", help="month in trips_clean.parquet")
    parser.add_argument("--current-file", type=Path, help="replay window parquet instead of --current month")
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "monitoring" / "reports")
    args = parser.parse_args()
    weather = load_weather(ROOT / "data" / "raw" / "weather_2026_01_07.json")
    zones = load_zone_categories(ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    trips = pd.read_parquet(args.trips)
    ref = month_frame(trips, args.reference, weather, zones)
    if args.current_file:
        cur, start, end = pd.read_parquet(args.current_file), "replay", "replay"
    else:
        cur, start, end = month_frame(trips, args.current, weather, zones), args.current, args.current
    result = compute_drift(ref, cur, start, end, html_path=args.out / "drift_report.html")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "drift_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
