import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from monitoring.drift import compute_drift
from monitoring.simulator import STORM_DURATION_FACTOR, replay

from features import WEATHER_COLUMNS
from serving import create_app

ZONES = list(range(1, 266))


def frame(seed, snow=0.0, factor=1.0, n=1500):
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({
        "hour": rng.integers(0, 24, n), "dow": rng.integers(0, 7, n),
        "temperature_2m": rng.normal(5, 3, n), "precipitation": rng.exponential(0.1, n),
        "snowfall": np.full(n, snow) + rng.exponential(0.01, n), "wind_speed_10m": rng.normal(12, 3, n),
        "duration_min": rng.normal(15, 4, n) * factor,
    })
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def test_no_drift_between_similar_windows():
    result = compute_drift(frame(0), frame(1), "a", "b")
    assert result["dataset_drift"] is False and result["drifted_features"] == []


def test_snowstorm_window_is_flagged_in_contract_format(tmp_path):
    result = compute_drift(frame(0), frame(2, snow=3.5, factor=1.4), "ref", "storm", html_path=tmp_path / "r.html")
    assert result["dataset_drift"] is True
    assert {"snowfall", "duration_min"} <= set(result["drifted_features"])
    assert set(result) == {"dataset_drift", "share_drifted_features", "drifted_features", "window_start", "window_end"}
    assert (tmp_path / "r.html").exists()
    json.dumps(result)  # serialisable for drift_report.json


class StubModel:
    version = "1"
    zone_categories = tuple(range(1, 266))

    def predict(self, features):
        return features["snowfall"].fillna(0) + 10.0


def test_simulator_replays_and_builds_storm_window(tmp_path):
    hours = pd.date_range("2026-04-01", periods=24 * 30, freq="h")
    weather = pd.DataFrame({"weather_hour": hours, **{c: 0.0 for c in WEATHER_COLUMNS}})
    rng = np.random.default_rng(0)
    trips = pd.DataFrame({
        "pickup_ts": pd.Timestamp("2026-04-01") + pd.to_timedelta(rng.integers(0, 28 * 24 * 60, 60), unit="m"),
        "PULocationID": rng.integers(1, 6, 60), "DOLocationID": rng.integers(1, 6, 60),
        "duration_min": 10.0, "month": "2026-04"})
    path = tmp_path / "missing.json"  # service falls back to NaN weather; overrides still apply
    with TestClient(create_app(StubModel, path)) as client:
        sent = []

        def send(payload):
            sent.append(payload)
            return client.post("/predict", json=payload)

        window = replay(trips, weather, ZONES, send, n=40, snowstorm=True)
        assert len(sent) == 40 and all("weather" in p for p in sent)
        assert "eta_requests_total{status=\"200\"} 40.0" in client.get("/metrics").text
    assert (window["snowfall"] == 3.5).all()
    assert (window["duration_min"] == 10.0 * STORM_DURATION_FACTOR).all()
