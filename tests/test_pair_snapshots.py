import numpy as np
import pandas as pd

from features.store import build_pair_snapshots


def trips(rows):
    return pd.DataFrame(
        {
            "pickup_ts": pd.to_datetime([r[0] for r in rows]),
            "PULocationID": [1] * len(rows),
            "DOLocationID": [2] * len(rows),
            "duration_min": [r[1] for r in rows],
        }
    )


def at(snaps, ts):
    row = snaps[snaps["event_timestamp"] == pd.Timestamp(ts, tz="UTC")]
    return row.iloc[0]


def test_trip_not_visible_before_it_completes():
    # picks up 08:50, 30 min long -> completes 09:20, lands in the 10:00 bucket
    snaps = build_pair_snapshots(trips([("2026-01-05 08:50", 30)]))
    assert snaps["event_timestamp"].min() == pd.Timestamp("2026-01-05 10:00", tz="UTC")
    assert at(snaps, "2026-01-05 10:00")["avg_duration_1h"] == 30


def test_one_hour_vs_24_hour_windows():
    snaps = build_pair_snapshots(trips([("2026-01-05 08:00", 10), ("2026-01-05 10:00", 30)]))
    # completions: 08:10 -> bucket 09:00, 10:30 -> bucket 11:00
    s = at(snaps, "2026-01-05 11:00")
    assert s["avg_duration_1h"] == 30
    assert s["avg_duration_24h"] == 20
    assert s["trip_count_24h"] == 2
    quiet = at(snaps, "2026-01-05 10:00")
    assert np.isnan(quiet["avg_duration_1h"])
    assert quiet["avg_duration_24h"] == 10


def test_trip_leaves_24h_window():
    snaps = build_pair_snapshots(trips([("2026-01-05 08:00", 10)]))
    # bucket 09:00; still counted through 09:00 + 23h, then an explicit empty-window row
    assert at(snaps, "2026-01-06 08:00")["trip_count_24h"] == 1
    last = at(snaps, "2026-01-06 09:00")
    assert snaps["event_timestamp"].max() == last["event_timestamp"]
    assert last["trip_count_24h"] == 0 and np.isnan(last["avg_duration_24h"])
