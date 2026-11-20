"""Traffic simulator: replay trips against /predict, optionally with an injected snowstorm.

Normal mode replays trips with their real weather. --snowstorm adds heavy snow to every
request (weather override) and inflates the ground-truth durations by 40%. The replayed
window (features + true duration) is saved so the drift job can compare it to a reference.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from features import WEATHER_COLUMNS, build_features, load_weather, load_zone_categories

ROOT = Path(__file__).resolve().parents[1]
STORM_WEATHER = {"temperature_2m": -4.0, "precipitation": 4.0, "snowfall": 3.5, "wind_speed_10m": 35.0}
STORM_DURATION_FACTOR = 1.4


def replay(trips: pd.DataFrame, weather: pd.DataFrame, zones: list[int], send, n: int = 500,
           snowstorm: bool = False, seed: int = 0, delay: float = 0.0) -> pd.DataFrame:
    """send(payload: dict) -> response-like with .status_code. Returns the replayed window."""
    sample = trips.sample(min(n, len(trips)), random_state=seed).sort_values("pickup_ts")
    for row in sample.itertuples():
        payload = {"pickup_zone": int(row.PULocationID), "dropoff_zone": int(row.DOLocationID),
                   "pickup_ts": pd.Timestamp(row.pickup_ts).isoformat()}
        if snowstorm:
            payload["weather"] = STORM_WEATHER
        send(payload)
        if delay:
            time.sleep(delay)

    window = build_features(sample, weather, zones)
    if snowstorm:
        for col in WEATHER_COLUMNS:
            window[col] = STORM_WEATHER[col]
        window["duration_min"] = window["duration_min"] * STORM_DURATION_FACTOR
    return window


if __name__ == "__main__":
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--month", default="2026-04")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    parser.add_argument("--snowstorm", action="store_true")
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "replay_window.parquet")
    args = parser.parse_args()
    trips = pd.read_parquet(args.trips)
    trips = trips[trips["month"] == args.month]
    with httpx.Client(base_url=args.url, timeout=10) as client:
        window = replay(trips, load_weather(ROOT / "data" / "raw" / "weather_2026_01_07.json"),
                        load_zone_categories(ROOT / "data" / "raw" / "taxi_zone_lookup.csv"),
                        lambda payload: client.post("/predict", json=payload), args.n, args.snowstorm,
                        delay=args.delay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(args.out, index=False)
    print(f"replayed {len(window)} requests; window saved to {args.out}")
