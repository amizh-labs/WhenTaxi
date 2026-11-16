"""Hourly rolling zone-pair duration snapshots (the Feast offline source).

Point-in-time rule: a trip contributes to a snapshot at time S only if it had
COMPLETED by S (pickup_ts + duration_min <= S). A request at time t therefore never
sees a trip that was still in progress, which is exactly what is known at serving time.

Naive NYC-local timestamps are labelled UTC so Feast can treat them as tz-aware;
the label is consistent everywhere so joins are unaffected.
"""
import numpy as np
import pandas as pd

ENTITY_KEY = "zone_pair"
FEATURE_NAMES = ["avg_duration_1h", "avg_duration_24h", "trip_count_24h"]
WINDOW_HOURS = 24


def pair_key(pu: pd.Series, do: pd.Series) -> pd.Series:
    return pu.astype("int64").astype(str) + "_" + do.astype("int64").astype(str)


def build_pair_snapshots(trips: pd.DataFrame) -> pd.DataFrame:
    """trips: cleaned table (pickup_ts, PULocationID, DOLocationID, duration_min).

    Returns columns: zone_pair, event_timestamp (hourly, UTC-labelled) and FEATURE_NAMES.
    Each pair gets a dense hourly grid from its first completion until 23h after its last,
    so the latest snapshot at or before any time is never stale. The final row (trip_count_24h
    = 0, NaN averages) records that the window emptied; the feature view has no TTL, so
    offline and online both keep returning that last row afterwards (no skew).
    """
    key = pair_key(trips["PULocationID"], trips["DOLocationID"])
    end = pd.to_datetime(trips["pickup_ts"]) + pd.to_timedelta(trips["duration_min"], unit="m")
    # Completion bucket S covers completions in (S-1h, S].
    done = pd.DataFrame({ENTITY_KEY: key, "bucket": end.dt.ceil("h"), "dur": trips["duration_min"]})
    hourly = done.groupby([ENTITY_KEY, "bucket"])["dur"].agg(["sum", "count"]).reset_index()

    frames = []
    for pair, grp in hourly.groupby(ENTITY_KEY):
        grid = pd.date_range(grp["bucket"].min(), grp["bucket"].max() + pd.Timedelta(hours=WINDOW_HOURS), freq="h")
        g = grp.set_index("bucket").reindex(grid, fill_value=0)
        sum_24 = g["sum"].rolling(WINDOW_HOURS, min_periods=1).sum()
        cnt_24 = g["count"].rolling(WINDOW_HOURS, min_periods=1).sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_1 = np.where(g["count"] > 0, g["sum"] / g["count"].where(g["count"] > 0), np.nan)
            avg_24 = np.where(cnt_24 > 0, sum_24 / cnt_24.where(cnt_24 > 0), np.nan)
        frames.append(
            pd.DataFrame(
                {
                    ENTITY_KEY: pair,
                    "event_timestamp": grid.tz_localize("UTC"),
                    "avg_duration_1h": avg_1.astype("float32"),
                    "avg_duration_24h": avg_24.astype("float32"),
                    "trip_count_24h": cnt_24.to_numpy().astype("float32"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
