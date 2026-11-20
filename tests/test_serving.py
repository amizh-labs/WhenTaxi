import json

import pytest
from fastapi.testclient import TestClient

from serving import create_app


class StubModel:
    version = "7"
    zone_categories = tuple(range(1, 266))

    def predict(self, features):
        # echo the snowfall so we can see the weather override reached the model
        return features["snowfall"].fillna(0) + 10.0 + features["hour"]


@pytest.fixture
def client(tmp_path):
    weather = {"hourly": {"time": ["2026-03-05T08:00"], "temperature_2m": [1.0], "precipitation": [0.0],
                          "snowfall": [0.5], "wind_speed_10m": [3.0]}}
    path = tmp_path / "weather.json"
    path.write_text(json.dumps(weather))
    with TestClient(create_app(StubModel, path)) as client:
        yield client


def body(**over):
    return {"pickup_zone": 161, "dropoff_zone": 237, "pickup_ts": "2026-03-05T08:30:00", **over}


def test_predict_contract_and_weather_from_file(client):
    resp = client.post("/predict", json=body())
    assert resp.status_code == 200
    assert resp.json() == {"eta_minutes": 18.5, "model_version": "7", "weather_source": "archive"}  # 0.5 snow + 10 + hour 8


def test_weather_override_for_snowstorm(client):
    resp = client.post("/predict", json=body(weather={"snowfall": 5.0}))
    assert resp.json()["eta_minutes"] == 23.0


def test_unknown_zone_and_missing_weather_hour_still_answer(client):
    resp = client.post("/predict", json=body(dropoff_zone=999, pickup_ts="2026-07-01T00:00:00"))
    assert resp.status_code == 200


@pytest.mark.parametrize("bad", [{"pickup_zone": 0}, {"pickup_zone": "x"}, {"pickup_ts": "not-a-date"}])
def test_invalid_input_rejected(client, bad):
    assert client.post("/predict", json=body(**bad)).status_code == 422


def test_metrics_exposed_with_contract_names(client):
    def count(status):
        for line in client.get("/metrics").text.splitlines():
            if line.startswith(f'eta_requests_total{{status="{status}"}}'):
                return float(line.split()[-1])
        return 0.0

    # Compare deltas: importing feast (other tests) puts prometheus_client in multiprocess
    # mode, where counters are shared across apps in one process. The API never imports feast.
    ok_before, bad_before = count("200"), count("422")
    client.post("/predict", json=body())
    client.post("/predict", json=body(pickup_zone=0))
    assert count("200") - ok_before == 1 and count("422") - bad_before == 1
    text = client.get("/metrics").text
    assert "eta_request_latency_seconds_bucket" in text
    assert "eta_prediction_minutes_bucket" in text


class RollingModel(StubModel):
    needs_rolling = True

    def predict(self, features):
        return features["avg_duration_24h"].fillna(-1) + features["trip_count_24h"]


class FakeProvider:
    def get(self, pickup_zone, dropoff_zone):
        if (pickup_zone, dropoff_zone) == (161, 237):
            return {"avg_duration_1h": 9.0, "avg_duration_24h": 12.0, "trip_count_24h": 5.0}
        return {"avg_duration_1h": float("nan"), "avg_duration_24h": float("nan"), "trip_count_24h": 0.0}


@pytest.mark.parametrize("provider", [FakeProvider, None])
def test_rolling_features_come_from_provider(tmp_path, provider):
    with TestClient(create_app(RollingModel, tmp_path / "none.json", provider and (lambda: provider()))) as client:
        known = client.post("/predict", json=body()).json()["eta_minutes"]
        unknown = client.post("/predict", json=body(dropoff_zone=5)).json()["eta_minutes"]
    assert known == (17.0 if provider else -1.0)  # 12 + 5 from the store; no provider -> NaN avg, count 0
    assert unknown == -1.0


def test_fault_injection_for_canary_demos(client, monkeypatch):
    normal = client.post("/predict", json=body()).json()["eta_minutes"]

    monkeypatch.setenv("ETA_FAULT", "bias")
    assert client.post("/predict", json=body()).json()["eta_minutes"] == pytest.approx(normal * 3)

    monkeypatch.setenv("ETA_FAULT", "errors")
    assert client.post("/predict", json=body()).status_code == 500

    monkeypatch.setenv("ETA_FAULT", "")
    assert client.post("/predict", json=body()).json()["eta_minutes"] == normal


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "7"}
