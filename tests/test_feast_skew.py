"""Training-serving skew test: offline (historical) and online feature values must match
for the same entity at the same timestamp. CI fails if they differ."""
import numpy as np
import pandas as pd
import pytest

from features.store import ENTITY_KEY, FEATURE_NAMES, FEATURE_REFS, build_pair_snapshots, make_store

SAMPLE = "tests/fixtures/trips_clean_sample.parquet"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("feast")
    snapshots = build_pair_snapshots(pd.read_parquet(SAMPLE))
    path = root / "snapshots.parquet"
    snapshots.to_parquet(path, index=False)
    fs, objects = make_store(root, path)
    fs.apply(objects)
    return fs, snapshots


def offline(fs, pairs, ts):
    entity_df = pd.DataFrame({ENTITY_KEY: pairs, "event_timestamp": pd.Timestamp(ts)})
    return fs.get_historical_features(entity_df=entity_df, features=FEATURE_REFS).to_df()


def online(fs, pairs):
    rows = [{ENTITY_KEY: p} for p in pairs]
    return pd.DataFrame(fs.get_online_features(features=FEATURE_REFS, entity_rows=rows).to_dict())


@pytest.mark.parametrize("checkpoint", ["2026-01-10 12:00", "2026-01-20 03:00", "2026-01-28 18:00"])
def test_offline_matches_online(store, checkpoint):
    fs, snapshots = store
    ts = pd.Timestamp(checkpoint, tz="UTC")
    pairs = sorted(snapshots.loc[snapshots["event_timestamp"] <= ts, ENTITY_KEY].unique())[:50]

    fs.materialize(start_date=snapshots["event_timestamp"].min().to_pydatetime(),
                   end_date=(ts + pd.Timedelta(seconds=1)).to_pydatetime())

    off = offline(fs, pairs, ts).set_index(ENTITY_KEY).loc[pairs, FEATURE_NAMES]
    on = online(fs, pairs).set_index(ENTITY_KEY).loc[pairs, FEATURE_NAMES]
    assert off.notna().to_numpy().any(), "test would be vacuous: no offline values found"
    np.testing.assert_allclose(off.to_numpy(float), on.to_numpy(float), rtol=1e-5, equal_nan=True)
