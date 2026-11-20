"""Build the online store the API reads: latest rolling zone-pair features as of a timestamp.

Run: python -m features.store.materialize [--upto "2026-03-31 23:00"]
Offline (training) and online (serving) values come from the same rolling_pair_features code,
and the skew test proves they match.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features.freshness import write_materialization_metadata
from features.rolling import ROLLING_COLUMNS, rolling_pair_features
from features.store.repo import FEATURE_REFS, make_store, open_store
from features.store.snapshots import ENTITY_KEY, pair_key

ROOT = Path(__file__).resolve().parents[3]
MAX_TRIP_HOURS = 3  # cleaning keeps durations <= 180 min, so older pickups cannot still matter


def latest_snapshot(trips: pd.DataFrame, upto: pd.Timestamp) -> pd.DataFrame:
    """One row per recently active zone pair with its features as of the hour boundary ``upto``."""
    upto = pd.Timestamp(upto).floor("h")
    recent = trips[(trips["pickup_ts"] > upto - pd.Timedelta(hours=24 + MAX_TRIP_HOURS))
                   & (trips["pickup_ts"] <= upto)]
    pairs = recent[["PULocationID", "DOLocationID"]].drop_duplicates().reset_index(drop=True)
    queries = pairs.assign(pickup_ts=upto)
    out = rolling_pair_features(recent, queries)
    out.insert(0, ENTITY_KEY, pair_key(pairs["PULocationID"], pairs["DOLocationID"]))
    out.insert(1, "event_timestamp", pd.Timestamp(upto).tz_localize("UTC"))
    return out[[ENTITY_KEY, "event_timestamp", *ROLLING_COLUMNS]]


def build_online_store(trips: pd.DataFrame, root: Path, upto: pd.Timestamp):
    """Write the snapshot, apply the Feast definitions and materialize into the online store."""
    root.mkdir(parents=True, exist_ok=True)
    snapshot = latest_snapshot(trips, upto)
    path = root / "snapshots.parquet"
    snapshot.to_parquet(path, index=False)
    store, objects = make_store(root, path)
    store.apply(objects)
    ts = snapshot["event_timestamp"].iloc[0].to_pydatetime()
    store.materialize(start_date=ts - pd.Timedelta(hours=1), end_date=ts + pd.Timedelta(seconds=1))
    write_materialization_metadata(root, upto)  # only after success: the API's eta_feature_age_seconds reads it
    return store


class FeastRollingProvider:
    """Serving-time lookup of rolling features from the online store (latest materialized values)."""

    def __init__(self, root: Path):
        self.store = open_store(root)

    def get(self, pickup_zone: int, dropoff_zone: int) -> dict[str, float]:
        key = f"{int(pickup_zone)}_{int(dropoff_zone)}"
        raw = self.store.get_online_features(features=FEATURE_REFS, entity_rows=[{ENTITY_KEY: key}]).to_dict()
        values = {name: raw[name][0] for name in ROLLING_COLUMNS}
        # A pair with no recent trips: training sees NaN averages and a count of 0.
        return {
            "avg_duration_1h": np.nan if values["avg_duration_1h"] is None else float(values["avg_duration_1h"]),
            "avg_duration_24h": np.nan if values["avg_duration_24h"] is None else float(values["avg_duration_24h"]),
            "trip_count_24h": 0.0 if values["trip_count_24h"] is None else float(values["trip_count_24h"]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "feast")
    parser.add_argument("--upto", default=None, help="default: hour after the last pickup in the data")
    args = parser.parse_args()
    data = pd.read_parquet(args.trips)
    upto = pd.Timestamp(args.upto) if args.upto else data["pickup_ts"].max().ceil("h")
    build_online_store(data, args.root, upto)
    print(f"online store materialized as of {upto} at {args.root}")
