import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from evaluation import Thresholds, decide, error_analysis, promote_if_better
from features import WEATHER_COLUMNS
from training import MODEL_NAME, train

ZONES = list(range(1, 266))


def comparison(champ_mae, chall_mae, champ_p90=10.0, chall_p90=10.0):
    return {"champion": {"mae": champ_mae, "p90_abs_error": champ_p90},
            "challenger": {"mae": chall_mae, "p90_abs_error": chall_p90}}


def test_decide_thresholds():
    assert decide(comparison(4.0, 3.5))[0] is True
    ok, reasons = decide(comparison(4.0, 3.99))  # only 0.25% better
    assert not ok and "MAE improvement" in reasons[0]
    assert decide(comparison(4.0, 4.5))[0] is False
    ok, reasons = decide(comparison(4.0, 3.5, chall_p90=10.5))  # better MAE but p90 +5%
    assert not ok and "p90" in reasons[0]
    assert decide(comparison(4.0, 3.5, chall_p90=10.1))[0] is True
    assert decide(comparison(4.0, 3.9), Thresholds(min_mae_improvement=0.05))[0] is False


def test_error_analysis_tables():
    n = 300
    scored = pd.DataFrame({
        "PULocationID": [1] * 150 + [2] * 150, "hour": [8, 9, 10] * 100,
        "duration_min": 10.0, "prediction": [10.0] * 150 + [14.0] * 150,
        "snowfall": [0.0] * 100 + [1.0] * 100 + [0.0] * 100,
        "precipitation": [0.0] * 200 + [2.0] * 100, "temperature_2m": 1.0})
    scored["abs_error"] = (scored["duration_min"] - scored["prediction"]).abs()
    tables = error_analysis(scored, min_zone_rows=100)
    assert tables["by_pickup_zone"].index.tolist() == [2, 1]  # worst zone first
    assert tables["by_pickup_zone"].loc[2, "mae"] == 4.0
    assert tables["by_pickup_zone"].loc[2, "bias"] == 4.0  # over-predicts
    assert set(tables["by_weather"].index) == {"clear", "snow", "rain"}
    assert tables["by_hour"].index.tolist() == [8, 9, 10] and tables["by_hour"]["rows"].sum() == n


def synthetic(months, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for month in months:
        n = 1500
        ts = pd.Timestamp(f"{month}-01") + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, n), unit="m")
        pu, do = rng.integers(1, 12, n), rng.integers(1, 12, n)
        frames.append(pd.DataFrame({"pickup_ts": ts, "PULocationID": pu, "DOLocationID": do,
                                    "duration_min": 5 + 3 * pu + do + rng.normal(0, 1, n), "month": month}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def weather():
    frame = pd.DataFrame({"weather_hour": pd.date_range("2026-01-01", periods=24 * 90, freq="h")})
    for col in WEATHER_COLUMNS:
        frame[col] = 0.0
    return frame


WEAK = {"n_estimators": 2, "num_leaves": 4, "learning_rate": 0.1}
STRONG = {"n_estimators": 80, "num_leaves": 16}


def run(weather, uri, first, second):
    trips = synthetic(["2026-01", "2026-02", "2026-03"])
    train(trips, weather, ZONES, uri, params=first)  # becomes champion (bootstrap)
    train(trips, weather, ZONES, uri, params=second)  # becomes challenger
    return promote_if_better(trips[trips["month"] == "2026-03"], weather, uri), MlflowClient(uri)


def test_better_challenger_is_promoted(weather, tmp_path):
    uri = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    report, client = run(weather, uri, WEAK, STRONG)
    assert report["promoted"] is True
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == 2
    assert client.get_model_version(MODEL_NAME, "2").tags["promotion"] == "promoted"
    names = {a.path for a in client.list_artifacts(client.get_model_version(MODEL_NAME, "2").run_id)}
    assert "evaluation_report.json" in names
    assert set(report["challenger_error_analysis"]) == {"by_pickup_zone", "by_hour", "by_weather"}


def test_worse_challenger_is_rejected_and_champion_unchanged(weather, tmp_path):
    uri = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    report, client = run(weather, uri, STRONG, WEAK)
    assert report["promoted"] is False and "MAE improvement" in report["reasons"][0]
    assert client.get_model_version_by_alias(MODEL_NAME, "champion").version == 1
    assert client.get_model_version(MODEL_NAME, "2").tags["promotion"] == "rejected"


def test_noop_when_challenger_is_champion(weather, tmp_path):
    uri = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    trips = synthetic(["2026-01", "2026-02", "2026-03"])
    train(trips, weather, ZONES, uri, params=STRONG)
    report = promote_if_better(trips[trips["month"] == "2026-03"], weather, uri)
    assert report["promoted"] is False and "already" in report["reasons"][0]
