"""Traffic simulator: replay trips against /predict as they are.

The replayed window (features + true duration) is returned so a drift job can compare it with a reference. For the
evening-rush scenario with a synthetic shift, see monitoring/surge_demo.py.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from features import build_features, load_weather, load_zone_categories

ROOT = Path(__file__).resolve().parents[1]


def replay(trips: pd.DataFrame, weather: pd.DataFrame, zones: list[int], send, n: int = 500,
           seed: int = 0, delay: float = 0.0) -> pd.DataFrame:
    """send(payload: dict) -> response-like with .status_code. Returns the replayed window."""
    sample = trips.sample(min(n, len(trips)), random_state=seed).sort_values("pickup_ts")
    for row in sample.itertuples():
        payload = {"pickup_zone": int(row.PULocationID), "dropoff_zone": int(row.DOLocationID),
                   "pickup_ts": pd.Timestamp(row.pickup_ts).isoformat()}
        send(payload)
        if delay:
            time.sleep(delay)
    return build_features(sample, weather, zones)


if __name__ == "__main__":
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--month", default="2026-03", help="a month present in trips_clean.parquet")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "replay_window.parquet")
    args = parser.parse_args()
    trips = pd.read_parquet(args.trips)
    trips = trips[trips["month"] == args.month]
    with httpx.Client(base_url=args.url, timeout=10) as client:
        window = replay(trips, load_weather(ROOT / "data" / "raw" / "weather_2026_01_07.json"),
                        load_zone_categories(ROOT / "data" / "raw" / "taxi_zone_lookup.csv"),
                        lambda payload: client.post("/predict", json=payload), args.n, delay=args.delay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    window.to_parquet(args.out, index=False)
    print(f"replayed {len(window)} requests; window saved to {args.out}")
