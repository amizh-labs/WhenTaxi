import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from features import WEATHER_COLUMNS, build_features, rolling_pair_features
from training import (
    MODEL_NAME,
    ZonePairMeanBaseline,
    default_tracking_uri,
    load_model,
    mae,
    p90_abs_error,
    rmse,
    train,
)

ZONES = list(range(1, 266))  # stands in for taxi_zone_lookup LocationIDs
FAST = {"n_estimators": 30, "num_leaves": 8, "learning_rate": 0.2}  # small, fast model for tests


def test_metrics_known_values():
    y, p = [10, 20, 30, 40], [12, 18, 30, 50]
    assert mae(y, p) == pytest.approx(3.5)
    assert rmse(y, p) == pytest.approx(np.sqrt((4 + 4 + 0 + 100) / 4))
    assert p90_abs_error(list(range(1, 11)), [0] * 10) == pytest.approx(9.1)


def test_baseline_pair_mean_and_fallbacks():
    trips = pd.DataFrame({"PULocationID": [1, 1, 2], "DOLocationID": [5, 5, 6],
                          "duration_min": [10.0, 20.0, 40.0]})
    base = ZonePairMeanBaseline().fit(trips)
    new = pd.DataFrame({"PULocationID": [1, 1, 9], "DOLocationID": [5, 7, 7]})
    # seen pair -> pair mean; unseen pair, seen pickup -> pickup mean; unseen both -> global mean
    assert base.predict(new).tolist() == [15.0, 15.0, pytest.approx(70 / 3)]


def synthetic(months):
    rng = np.random.default_rng(0)
    frames = []
    for month in months:
        n = 400
        ts = pd.Timestamp(f"{month}-01") + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, n), unit="m")
        pu, do = rng.integers(1, 6, n), rng.integers(1, 6, n)
        frames.append(pd.DataFrame({"pickup_ts": ts, "PULocationID": pu, "DOLocationID": do,
                                    "duration_min": 5 + 3 * pu + do + rng.normal(0, 1, n), "month": month}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def weather():
    frame = pd.DataFrame({"weather_hour": pd.date_range("2026-01-01", periods=24 * 90, freq="h")})
    for col in WEATHER_COLUMNS:
        frame[col] = 0.0
    return frame


@pytest.fixture
def uri(tmp_path):
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_train_registers_model_with_aliases(weather, uri):
    trips = synthetic(["2026-01", "2026-02", "2026-03"])
    result = train(trips, weather, ZONES, uri, params=FAST)

    assert result["metrics"]["mae"] < result["metrics"]["baseline_mae"] * 1.5
    assert {"mae", "rmse", "p90_abs_error", "baseline_mae"} <= result["metrics"].keys()
    client = MlflowClient(uri)
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == int(result["version"])

    second = train(trips, weather, ZONES, uri, params=FAST)
    # champion stays on version 1; only the challenger moves
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == int(result["version"])
    assert client.get_model_version_by_alias(MODEL_NAME, "challenger").version == int(second["version"])


def test_everything_is_logged_to_one_run(weather, uri):
    report = {"month": "2026-01", "rows_in": 10, "rows_out": 9}
    result = train(synthetic(["2026-01", "2026-02", "2026-03"]), weather, ZONES, uri,
                   clean_report=report, params=FAST)
    client = MlflowClient(uri)
    run = client.get_run(result["run_id"])
    assert "baseline_mae" in run.data.metrics and "mae" in run.data.metrics
    assert run.data.tags["train_months"] == "2026-01,2026-02" and run.data.tags["valid_months"] == "2026-03"
    assert set(run.data.metrics) >= {f"valid_nan_rate_{c}" for c in WEATHER_COLUMNS}
    names = {a.path for a in client.list_artifacts(result["run_id"])}
    assert {"feature_importance.json", "clean_report.json", "zone_categories.json"} <= names


def test_unseen_zone_does_not_crash_and_categories_roundtrip(weather, uri):
    train(synthetic(["2026-01", "2026-02", "2026-03"]), weather, ZONES, uri, params=FAST)
    loaded = load_model("champion", uri)
    assert loaded.zone_categories == ZONES  # saved with the model, reused at serving

    trips = pd.DataFrame({
        "pickup_ts": pd.to_datetime(["2026-03-05 08:00"] * 3),
        "PULocationID": [1, 200, 999],  # trained / valid-but-never-seen / not a zone at all
        "DOLocationID": [2, 201, 999],
    })
    assert loaded.needs_rolling  # default training uses the rolling zone-pair features
    history = synthetic(["2026-01", "2026-02", "2026-03"])
    features = build_features(trips, weather, loaded.zone_categories, rolling_pair_features(history, trips))
    assert features["PULocationID"].isna().tolist() == [False, False, True]
    preds = loaded.predict(features)
    assert np.isfinite(preds).all() and len(preds) == 3


def test_rolling_features_can_be_turned_off(weather, uri):
    train(synthetic(["2026-01", "2026-02", "2026-03"]), weather, ZONES, uri, params=FAST, use_rolling=False)
    assert not load_model("champion", uri).needs_rolling


def test_tracking_uri_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example:5000")
    assert default_tracking_uri() == "http://mlflow.example:5000"
    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    assert default_tracking_uri().startswith("sqlite:///") and default_tracking_uri().endswith("mlflow.db")


def test_split_is_by_time_and_empty_split_fails(weather, uri):
    with pytest.raises(ValueError):
        train(synthetic(["2026-01", "2026-02"]), weather, ZONES, uri)


def test_geo_and_log_target_model_round_trips_in_minutes(weather, uri):
    trips = synthetic(["2026-01", "2026-02", "2026-03"])
    result = train(trips, weather, ZONES, uri, params=FAST, use_geo=True, log_target=True)
    loaded = load_model("champion", tracking_uri=uri)
    assert loaded.needs_geo and loaded.log_target and "distance_km" in loaded.feature_columns
    rows = trips.iloc[:5]
    feats = build_features(rows, weather, loaded.zone_categories, rolling_pair_features(trips, rows), geo=True)
    pred = loaded.predict(feats)
    assert pred.between(0, 200).all()  # expm1 applied: minutes, not log-minutes
    assert result["metrics"]["mae"] < 15


def test_plain_model_does_not_need_geo(weather, uri):
    train(synthetic(["2026-01", "2026-02", "2026-03"]), weather, ZONES, uri, params=FAST)
    loaded = load_model("champion", tracking_uri=uri)
    assert not loaded.needs_geo and not loaded.log_target
