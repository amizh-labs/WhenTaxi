import json

import numpy as np
import pandas as pd
import pytest
from flows.retrain import read_drift, should_retrain
from flows.training_flow import retrain_flow, training_flow
from prefect.testing.utilities import prefect_test_harness


def test_should_retrain_decisions():
    used = ["2026-01", "2026-02", "2026-03"]
    assert should_retrain(used + ["2026-04"], used, None) == (True, "new data: 2026-04")
    assert should_retrain(used, used, {"dataset_drift": True, "share_drifted_features": 0.4})[0] is True
    assert should_retrain(used, used, {"dataset_drift": False})[0] is False
    assert should_retrain(used, used, None)[0] is False


def test_read_drift_missing_and_present(tmp_path):
    assert read_drift(None) is None and read_drift(tmp_path / "nope.json") is None
    path = tmp_path / "drift_report.json"
    path.write_text(json.dumps({"dataset_drift": True}))
    assert read_drift(path) == {"dataset_drift": True}


def make_raw(raw_dir, months):
    """Tiny raw dataset: real column names/layout, zones 1-5, 200 trips a month."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"LocationID": range(1, 6), "Borough": "Manhattan", "Zone": "Z", "service_zone": "Yellow Zone"}
                 ).to_csv(raw_dir / "taxi_zone_lookup.csv", index=False)
    hours = pd.date_range("2026-01-01", periods=24 * 150, freq="h")
    hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours]}
    for col in ("temperature_2m", "precipitation", "snowfall", "wind_speed_10m"):
        hourly[col] = [0.0] * len(hours)
    (raw_dir / "weather_2026_01_07.json").write_text(json.dumps({"hourly": hourly}))
    rng = np.random.default_rng(0)
    for month in months:
        pickup = pd.Timestamp(f"{month}-01") + pd.to_timedelta(rng.integers(0, 27 * 24 * 60, 200), unit="m")
        pd.DataFrame({
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(rng.integers(5, 40, 200), unit="m"),
            "PULocationID": rng.integers(1, 6, 200), "DOLocationID": rng.integers(1, 6, 200),
        }).to_parquet(raw_dir / f"yellow_tripdata_{month}.parquet")


@pytest.fixture(scope="module")
def harness():
    with prefect_test_harness():
        yield


def test_training_then_retrain_on_new_month(harness, tmp_path):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    make_raw(raw, ["2026-01", "2026-02", "2026-03"])

    feast = tmp_path / "feast"
    first = training_flow(["2026-01", "2026-02", "2026-03"], raw, processed, uri, feast_root=feast)
    assert first["version"] == "1" and (processed / "trips_clean.parquet").exists()
    assert (feast / "online.db").exists() and first["online_features_as_of"]

    # nothing new -> no retrain
    assert retrain_flow(None, raw, processed, uri, feast_root=feast)["retrained"] is False

    # drift flag alone triggers retrain
    drift = tmp_path / "drift_report.json"
    drift.write_text(json.dumps({"dataset_drift": True, "share_drifted_features": 0.5}))
    assert retrain_flow(drift, raw, processed, uri, feast_root=feast)["retrained"] is True

    # new month triggers retrain and trains on earlier months, validating on the new one
    make_raw(raw, ["2026-01", "2026-02", "2026-03", "2026-04"])
    out = retrain_flow(None, raw, processed, uri, feast_root=feast)
    assert out["retrained"] and "2026-04" in out["reason"]


def test_daily_drift_flow_on_an_empty_log_reports_insufficient_data(tmp_path):
    from flows.drift_flow import CRON, drift_flow

    with prefect_test_harness():
        out = drift_flow(tmp_path / "no_log.sqlite", tmp_path / "reports", retrain=True)
    assert out["report"]["status"] == "insufficient data" and out["report"]["dataset_drift"] is False
    assert out["retrain"] is None  # nothing drifted, so the retrain flow is not even asked
    assert json.loads((tmp_path / "reports" / "drift_report.json").read_text())["status"] == "insufficient data"
    assert CRON == "0 6 * * *"


def test_daily_drift_flow_hands_a_drifted_window_to_the_retrain_flow(tmp_path, monkeypatch):
    import flows.drift_flow as module
    import flows.training_flow as training

    drifted = {"status": "ok", "dataset_drift": True, "drifted_columns": ["distance_km"], "context_drift": {
        "drifted_columns": []}, "data_quality": {"share_missing_weather": 0.0, "alert": False}}
    monkeypatch.setattr(module, "run_drift", lambda *a, **k: drifted)
    seen = []
    monkeypatch.setattr(training, "retrain_flow", lambda report: seen.append(report) or {"retrained": True})
    with prefect_test_harness():
        out = module.drift_flow(tmp_path / "log.sqlite", tmp_path / "reports", retrain=True)
        without = module.drift_flow(tmp_path / "log.sqlite", tmp_path / "reports", retrain=False)
    assert out["retrain"] == {"retrained": True} and seen == [tmp_path / "reports" / "drift_report.json"]
    assert without["retrain"] is None


def test_training_flow_trains_the_shipped_geo_l1_config(monkeypatch, tmp_path):
    from flows import training_flow as tf

    seen = {}

    def fake_train(*args, **kwargs):
        seen.update(kwargs)
        return {"run_id": "r", "version": "1", "metrics": {"mae": 1.0, "baseline_mae": 2.0}}

    monkeypatch.setattr(tf, "train", fake_train)
    monkeypatch.setattr(tf.pd, "read_parquet", lambda p: None)
    monkeypatch.setattr(tf, "load_weather", lambda p: None)
    monkeypatch.setattr(tf, "load_zone_categories", lambda p: None)
    out = tf.train_task.fn(tmp_path / "t.parquet", tmp_path, [], ["2026-01"], ["2026-02"], "uri", None)
    assert out["version"] == "1"
    assert seen["use_geo"] is True and seen["params"] == {"objective": "l1"}
