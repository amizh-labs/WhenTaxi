import numpy as np
import pandas as pd

from features import ROLLING_COLUMNS, build_features, rolling_pair_features
from features.store import FeastRollingProvider, build_online_store, build_pair_snapshots, latest_snapshot, pair_key


def trips(rows):
    return pd.DataFrame({"pickup_ts": pd.to_datetime([r[0] for r in rows]), "PULocationID": 1,
                         "DOLocationID": 2, "duration_min": [r[1] for r in rows]})


def query(ts):
    return pd.DataFrame({"pickup_ts": pd.to_datetime([ts]), "PULocationID": [1], "DOLocationID": [2]})


def test_trip_invisible_until_completed():
    history = trips([("2026-01-05 08:50", 30)])  # completes 09:20
    assert rolling_pair_features(history, query("2026-01-05 09:30"))["trip_count_24h"].iloc[0] == 0  # hour 09:00
    assert rolling_pair_features(history, query("2026-01-05 10:05"))["trip_count_24h"].iloc[0] == 1


def test_windows_and_empty_defaults():
    history = trips([("2026-01-05 08:00", 10), ("2026-01-05 10:00", 30)])
    got = rolling_pair_features(history, query("2026-01-05 11:20")).iloc[0]
    assert got["avg_duration_1h"] == 30 and got["avg_duration_24h"] == 20 and got["trip_count_24h"] == 2
    quiet = rolling_pair_features(history, query("2026-01-05 10:20")).iloc[0]
    assert np.isnan(quiet["avg_duration_1h"]) and quiet["avg_duration_24h"] == 10
    later = rolling_pair_features(history, query("2026-01-07 00:00")).iloc[0]
    assert np.isnan(later["avg_duration_24h"]) and later["trip_count_24h"] == 0


def test_build_features_appends_rolling_columns():
    weather = pd.DataFrame({"weather_hour": pd.to_datetime(["2026-01-05 09:00"]), "temperature_2m": 0.0,
                            "precipitation": 0.0, "snowfall": 0.0, "wind_speed_10m": 0.0})
    q = query("2026-01-05 09:30")
    rolling = rolling_pair_features(trips([("2026-01-05 08:00", 10)]), q)
    out = build_features(q, weather, rolling=rolling)
    assert list(out.columns[-3:]) == ROLLING_COLUMNS


def sample_with_repeats():
    frame = pd.read_parquet("tests/fixtures/trips_clean_sample.parquet")
    rng = np.random.default_rng(1)
    frame["PULocationID"], frame["DOLocationID"] = rng.integers(1, 4, len(frame)), rng.integers(1, 4, len(frame))
    return frame


def test_matches_feast_offline_snapshots_exactly():
    """Training features (vectorised) == the Feast snapshots that feed the stores."""
    history = sample_with_repeats()
    snaps = build_pair_snapshots(history)
    snaps["event_timestamp"] = snaps["event_timestamp"].dt.tz_localize(None)
    queries = history.assign(pickup_ts=history["pickup_ts"] + pd.Timedelta(hours=30)).reset_index(drop=True)
    mine = rolling_pair_features(history, queries)[ROLLING_COLUMNS].to_numpy(float)

    q = queries.assign(zone_pair=pair_key(queries["PULocationID"], queries["DOLocationID"]),
                       H=queries["pickup_ts"].dt.floor("h"))
    ref = q.merge(snaps, how="left", left_on=["zone_pair", "H"], right_on=["zone_pair", "event_timestamp"])
    expected = ref[ROLLING_COLUMNS].fillna({"trip_count_24h": 0}).to_numpy(float)
    assert (np.isnan(mine) == np.isnan(expected)).all()
    np.testing.assert_allclose(np.nan_to_num(mine), np.nan_to_num(expected), rtol=1e-5)


def test_online_store_serves_same_values_as_training_code(tmp_path):
    """Serving (Feast online store) == training (rolling_pair_features) for the same pair and time."""
    history = sample_with_repeats()
    upto = history["pickup_ts"].max().ceil("h") + pd.Timedelta(hours=1)
    snapshot = latest_snapshot(history, upto)
    build_online_store(history, tmp_path, upto)
    provider = FeastRollingProvider(tmp_path)

    checked = 0
    for pu in (1, 2, 3):
        for do in (1, 2, 3):
            expected = rolling_pair_features(history, pd.DataFrame(
                {"pickup_ts": [upto], "PULocationID": [pu], "DOLocationID": [do]})).iloc[0]
            got = provider.get(pu, do)
            for col in ROLLING_COLUMNS:
                assert (np.isnan(got[col]) and np.isnan(expected[col])) or np.isclose(got[col], expected[col], rtol=1e-5)
            checked += 1
    assert checked == 9 and len(snapshot) > 0
    assert provider.get(200, 201)["trip_count_24h"] == 0.0  # pair never seen: count 0, NaN averages
