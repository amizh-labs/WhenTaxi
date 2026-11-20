"""Stage L1: live weather in /predict. Every HTTP call is mocked; the autouse guard in conftest blocks the real one."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from features import WEATHER_COLUMNS
from serving import create_app
from serving.live_weather import FAILURE_RETRY_SECONDS, PARAMS, URL, LiveWeather, parse_hourly

NYC = "America/New_York"
NOW = pd.Timestamp("2026-11-18 14:20", tz=NYC).timestamp()  # 'now' in these tests


def payload(day="2026-11-18", snowfall=2.0):
    times = pd.date_range(f"{day} 00:00", periods=24 * 3, freq="h")
    return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                       "temperature_2m": [5.0] * len(times), "precipitation": [0.3] * len(times),
                       "snowfall": [snowfall] * len(times), "wind_speed_10m": [11.0] * len(times)}}


class Fetch:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result if result is not None else payload(), error, []

    def __call__(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        if self.error:
            raise self.error
        return self.result


def live(fetch, now=NOW):
    clock = {"t": now}
    return LiveWeather(fetch, clock=lambda: clock["t"]), clock


# ---- the request and the lookup ---------------------------------------------------------------------------------

def test_request_uses_hourly_forecast_not_current_block_with_default_units():
    fetch = Fetch()
    provider, _ = live(fetch)
    assert provider.lookup(pd.Timestamp("2026-11-18 14:35")) == {
        "temperature_2m": 5.0, "precipitation": 0.3, "snowfall": 2.0, "wind_speed_10m": 11.0}
    (url, params, timeout), = fetch.calls
    assert url == URL and url.endswith("/v1/forecast")
    assert params["hourly"].split(",") == WEATHER_COLUMNS and params["timezone"] == "America/New_York"
    assert "current" not in params and not any("unit" in k for k in params)  # default units, like the archive file
    assert timeout == 2.0 and params == PARAMS


def test_row_is_picked_by_pickup_hour_and_can_be_a_future_hour():
    data = payload()
    data["hourly"]["temperature_2m"] = [float(i) for i in range(len(data["hourly"]["time"]))]
    provider, _ = live(Fetch(data))
    assert provider.lookup(pd.Timestamp("2026-11-18 14:59"))["temperature_2m"] == 14.0
    assert provider.lookup(pd.Timestamp("2026-11-18 16:00"))["temperature_2m"] == 16.0  # an hour or two ahead


def test_cached_per_clock_hour():
    fetch = Fetch()
    provider, clock = live(fetch)
    for _ in range(3):
        provider.lookup(pd.Timestamp("2026-11-18 14:35"))
    assert len(fetch.calls) == 1
    clock["t"] += 3600  # next clock hour: refresh
    provider.lookup(pd.Timestamp("2026-11-18 15:35"))
    assert len(fetch.calls) == 2


# ---- failures return None, never raise --------------------------------------------------------------------------

@pytest.mark.parametrize("error", [TimeoutError("2s"), ConnectionError("down"), OSError("dns")])
def test_network_failure_returns_none_and_backs_off(error):
    fetch = Fetch(error=error)
    provider, clock = live(fetch)
    assert provider.lookup(pd.Timestamp("2026-11-18 14:35")) is None
    assert provider.lookup(pd.Timestamp("2026-11-18 14:35")) is None
    assert len(fetch.calls) == 1  # an outage costs one timeout, not one per request
    clock["t"] += FAILURE_RETRY_SECONDS + 1
    provider.lookup(pd.Timestamp("2026-11-18 14:35"))
    assert len(fetch.calls) == 2


@pytest.mark.parametrize("bad", [{}, {"hourly": {}}, {"hourly": {"time": []}}, {"error": True, "reason": "x"},
                                 {"hourly": {"time": ["2026-11-18T14:00"], "temperature_2m": [1.0]}},
                                 {"hourly": {"time": ["2026-11-18T14:00"], **{c: ["warm"] for c in WEATHER_COLUMNS}}}])
def test_malformed_response_returns_none(bad):
    provider, _ = live(Fetch(bad))
    assert provider.lookup(pd.Timestamp("2026-11-18 14:35")) is None


def test_null_values_for_the_hour_return_none():
    data = payload()
    data["hourly"]["snowfall"][14] = None
    provider, _ = live(Fetch(data))
    assert provider.lookup(pd.Timestamp("2026-11-18 14:35")) is None


@pytest.mark.parametrize("ts", ["2026-01-15 08:00", "2026-07-31 23:00", "2026-11-12 09:00", "2026-11-30 09:00"])
def test_timestamps_outside_the_forecast_window_never_call_the_network(ts):
    fetch = Fetch()
    provider, _ = live(fetch)
    assert provider.lookup(pd.Timestamp(ts)) is None
    assert fetch.calls == []


def test_parse_hourly_rejects_garbage():
    with pytest.raises((KeyError, ValueError, TypeError)):
        parse_hourly({"hourly": {"time": ["x"]}})


# ---- /predict: source order request > live > archive > missing ----------------------------------------------------

class StubModel:
    version = "7"
    zone_categories = tuple(range(1, 266))

    def predict(self, features):  # echoes the snowfall (-100 when NaN) and the pickup hour
        return features["snowfall"].fillna(-100) + 10.0 + features["hour"]


class StubLive:
    def __init__(self, values=None):
        self.values, self.asked = values, []

    def lookup(self, ts):
        self.asked.append(ts)
        return self.values


ARCHIVE = {"hourly": {"time": ["2026-03-05T08:00"], "temperature_2m": [1.0], "precipitation": [0.0],
                      "snowfall": [0.5], "wind_speed_10m": [3.0]}}


def make_client(tmp_path, stub):
    path = tmp_path / "weather.json"
    path.write_text(json.dumps(ARCHIVE))
    return TestClient(create_app(StubModel, path, rolling_provider=None, live_weather=lambda: stub))


def post(client, ts="2026-11-18T14:30:00", **extra):
    resp = client.post("/predict", json={"pickup_zone": 161, "dropoff_zone": 237, "pickup_ts": ts, **extra})
    assert resp.status_code == 200
    return resp.json()


LIVE = {"temperature_2m": 5.0, "precipitation": 0.3, "snowfall": 2.0, "wind_speed_10m": 11.0}


def test_live_weather_reaches_the_model(tmp_path):
    with make_client(tmp_path, StubLive(LIVE)) as client:
        out = post(client)
    assert out["weather_source"] == "live" and out["eta_minutes"] == 2.0 + 10 + 14


def test_archive_when_live_has_nothing(tmp_path):
    with make_client(tmp_path, StubLive(None)) as client:
        out = post(client, ts="2026-03-05T08:30:00")
    assert out["weather_source"] == "archive" and out["eta_minutes"] == 0.5 + 10 + 8


def test_november_2026_without_any_coverage_is_missing_not_a_crash(tmp_path):
    with make_client(tmp_path, StubLive(None)) as client:
        out = post(client, ts="2026-11-18T14:30:00")
    assert out["weather_source"] == "missing" and out["eta_minutes"] == -100 + 10 + 14  # NaN weather reached the model


def test_request_weather_wins_and_skips_the_network(tmp_path):
    stub = StubLive(LIVE)
    full = {"temperature_2m": 0.0, "precipitation": 9.0, "snowfall": 7.0, "wind_speed_10m": 1.0}
    with make_client(tmp_path, stub) as client:
        out = post(client, weather=full)
    assert out["weather_source"] == "request" and out["eta_minutes"] == 7.0 + 10 + 14
    assert stub.asked == []


def test_partial_request_weather_overrides_only_what_it_sets(tmp_path):
    with make_client(tmp_path, StubLive(LIVE)) as client:
        out = post(client, weather={"snowfall": 5.0})
    assert out["weather_source"] == "request" and out["eta_minutes"] == 5.0 + 10 + 14


def test_no_live_provider_falls_back_to_archive_then_missing(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(ARCHIVE))
    with TestClient(create_app(StubModel, path, rolling_provider=None, live_weather=None)) as client:
        assert post(client, ts="2026-03-05T08:10:00")["weather_source"] == "archive"
        assert post(client, ts="2026-09-05T08:10:00")["weather_source"] == "missing"


def test_real_network_is_blocked_in_tests():
    with pytest.raises(BaseException, match="real network"):
        LiveWeather()._fetch(URL, PARAMS, 2.0)  # the default fetcher is the (blocked) real one
