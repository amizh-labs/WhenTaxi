"""The evening-rush surge demo (monitoring/surge_demo.py): fixtures and mocked HTTP only, never the real network,
the real registry, the real Feast store or Docker."""
import json
import os
import socket
import subprocess
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from monitoring import surge_demo as sd
from monitoring.prediction_drift import reference_features
from monitoring.surge_demo import (
    AIRPORT_ZONES,
    ComposeLog,
    DemoConfig,
    DemoError,
    LogError,
    assert_throwaway_store,
    build_history,
    composition,
    format_summary,
    httpx_sender,
    mae_flag,
    replay,
    retrain_challenger,
    run_demo,
    sample_baseline,
    sample_surge,
    score,
)

from features import WEATHER_COLUMNS, rolling_pair_features
from serving.prediction_log import COLUMNS

N = 600  # requests per window; the drift check needs at least MIN_ROWS (200) logged rows
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_sockets(monkeypatch):
    """Nothing here may open a connection: the API is a httpx MockTransport, Docker and MLflow are fakes."""
    def blocked(*args, **kwargs):
        raise AssertionError("a test tried to open a real network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)


# ---- fixtures: synthetic cleaned trips, weather, a fake API and a fake log -------------------------------------------

# Mostly central-Manhattan zones, like real yellow-taxi traffic: most trips are short, so airport trips stand out.
CORE_ZONES = np.array([43, 48, 50, 68, 79, 90, 100, 107, 113, 114, 125, 137, 141, 142, 143, 144, 148, 151, 158, 161,
                       162, 163, 164, 170, 186, 230, 231, 233, 234, 236, 237, 238, 239, 246, 249])


def zones(rng, n):
    return np.where(rng.random(n) < 0.95, rng.choice(CORE_ZONES, n), rng.integers(1, 264, n))


def make_trips(n, seed, start, days, month, duration=None):
    rng = np.random.default_rng(seed)
    minutes = rng.integers(0, days * 1440, n)
    pickup, dropoff = zones(rng, n), zones(rng, n)
    airport = rng.random(n) < 0.04  # about 4% of trips start or end at an airport
    from_airport = rng.random(n) < 0.5
    pickup = np.where(airport & from_airport, rng.choice(AIRPORT_ZONES, n), pickup)
    dropoff = np.where(airport & ~from_airport, rng.choice(AIRPORT_ZONES, n), dropoff)
    return pd.DataFrame({
        "pickup_ts": pd.Timestamp(start) + pd.to_timedelta(minutes, unit="m"),
        "PULocationID": pickup, "DOLocationID": dropoff,
        "duration_min": rng.uniform(5, 40, n) if duration is None else float(duration), "month": month})


@pytest.fixture(scope="module")
def weather():
    hours = pd.date_range("2026-01-01", "2026-04-30 23:00", freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame({"weather_hour": hours, "temperature_2m": np.linspace(-5, 18, len(hours)),  # warming up: context
                         "precipitation": rng.exponential(0.05, len(hours)), "snowfall": 0.0,
                         "wind_speed_10m": rng.normal(12, 2, len(hours))})


@pytest.fixture(scope="module")
def april():
    return make_trips(30_000, 10, "2026-04-01", 30, "2026-04")


@pytest.fixture(scope="module")
def march_tail():
    return make_trips(3_000, 11, "2026-03-29", 3, "2026-03")


@pytest.fixture(scope="module")
def reference(weather):
    jan_feb = pd.concat([make_trips(15_000, 12, "2026-01-01", 31, "2026-01"),
                         make_trips(15_000, 13, "2026-02-01", 28, "2026-02")], ignore_index=True)
    return reference_features(jan_feb, weather)


class ConstantModel:
    """Predicts a fixed ETA; no rolling or geo features, so it needs no store."""
    version = "1"
    zone_categories = tuple(range(1, 266))
    needs_rolling = False
    needs_geo = False

    def __init__(self, value=10.0):
        self.value = value

    def predict(self, features):
        return pd.Series(self.value, index=features.index)


class FakeApi:
    """A mocked /predict behind httpx.MockTransport. Only successful requests reach `logged` (the API logs those)."""

    def __init__(self, fail=None):
        self.fail, self.logged, self.calls = fail, [], 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.fail is not None:
            outcome = self.fail(self.calls)
            if isinstance(outcome, Exception):
                raise outcome
            if outcome:
                return httpx.Response(outcome)
        self.logged.append(json.loads(request.content))
        return httpx.Response(200, json={"eta_minutes": 15.0, "model_version": "1", "weather_source": "archive"})

    def sender(self):
        return httpx_sender(httpx.Client(transport=httpx.MockTransport(self.handler), base_url="http://api.test"))


class FakeLog:
    """What the compose volume would hold: one row per logged request, weather looked up like the archive path."""

    def __init__(self, api, weather, reset_error=None):
        self.api, self.weather, self.reset_error, self.resets, self.fetches = api, weather, reset_error, 0, []

    def reset(self):
        if self.reset_error:
            raise LogError(self.reset_error)
        self.resets += 1
        self.api.logged.clear()

    def fetch(self, name):
        self.fetches.append(name)
        rows = pd.DataFrame(self.api.logged, columns=["pickup_zone", "dropoff_zone", "pickup_ts"])
        if rows.empty:
            return pd.DataFrame(columns=COLUMNS)
        hour = pd.to_datetime(rows["pickup_ts"]).dt.floor("h")
        wx = self.weather.drop_duplicates("weather_hour").set_index("weather_hour").reindex(hour)
        out = rows.copy()
        for col in WEATHER_COLUMNS:
            out[col] = wx[col].to_numpy()
        out["ts"] = pd.date_range("2026-11-25 09:00", periods=len(out), freq="s").astype(str)
        out["weather_source"], out["eta_minutes"], out["model_version"] = "archive", 15.0, "1"
        return out[COLUMNS]


def demo_cfg(tmp_path, **kw):
    return DemoConfig(n=N, seed=0, reference_rows=3000, work=tmp_path / "work", report=tmp_path / "report.json", **kw)


def run(tmp_path, april, march_tail, weather, reference, *, api=None, log=None, retrain=None, model=None, cfg=None,
        champion_mae=3.79):
    api = api or FakeApi()
    log = log or FakeLog(api, weather)
    calls = []

    def fake_retrain(drift_path):
        calls.append(drift_path)
        return {"triggered": True, "reason": "drift detected", "challenger_version": "2"}

    report = run_demo(cfg or demo_cfg(tmp_path), april=april, march=march_tail, reference=reference, weather=weather,
                      send=api.sender(), log=log, champion=lambda: (model or ConstantModel(), champion_mae),
                      retrain=retrain or fake_retrain)
    return report, calls, api, log


@pytest.fixture(scope="module")
def demo(tmp_path_factory, april, march_tail, weather, reference):
    tmp = tmp_path_factory.mktemp("demo")
    return run(tmp, april, march_tail, weather, reference) + (tmp,)


# ---- 1. the sampler really shifts hours and the airport share ----------------------------------------------------------

def test_surge_sampler_shifts_hours_and_airport_share(april):
    base, surge = sample_baseline(april, 2000, seed=0), sample_surge(april, 2000, seed=0)
    assert len(base) == len(surge) == 2000
    b, s = composition(base), composition(surge)
    assert 0.20 < b["evening_share"] < 0.30  # 6 of 24 hours, uniform in the fixture
    assert b["airport_share"] < 0.06
    assert s["evening_share"] == pytest.approx(0.85, abs=0.01)  # resampled toward 16-21h
    assert s["airport_share"] == pytest.approx(0.30, abs=0.02)  # many more airport trips than the baseline
    assert s["airport_share"] > 5 * b["airport_share"]
    assert surge["pickup_ts"].is_monotonic_increasing and base["pickup_ts"].is_monotonic_increasing
    assert set(AIRPORT_ZONES) & (set(surge["PULocationID"]) | set(surge["DOLocationID"]))


def test_both_samplers_keep_the_months_weekday_mix(april):
    month = april["pickup_ts"].dt.dayofweek.value_counts(normalize=True).sort_index()
    for window in (sample_baseline(april, 2000, seed=3), sample_surge(april, 2000, seed=3)):
        mix = window["pickup_ts"].dt.dayofweek.value_counts(normalize=True).sort_index()
        assert (0.5 * (mix - month).abs().sum()) < 0.02  # only hour and airport share are shifted, never the weekday


def test_sampler_draws_with_replacement_from_a_small_pool_and_fails_clearly_on_an_empty_one(april):
    tiny = april[april["PULocationID"].isin(AIRPORT_ZONES) | (april["pickup_ts"].dt.hour < 5)].head(40)
    assert len(sample_baseline(tiny, 100)) == 100  # fewer trips than asked: drawn with replacement
    no_airport = april[~(april["PULocationID"].isin(AIRPORT_ZONES) | april["DOLocationID"].isin(AIRPORT_ZONES))]
    with pytest.raises(DemoError, match="none of that kind"):
        sample_surge(no_airport, 100)


# ---- 2. drift: baseline quiet, surge flags hour and distance only (weather is context) ------------------------------

def test_baseline_window_shows_no_drift(demo):
    baseline = demo[0]["windows"][0]
    assert baseline["dataset_drift"] is False and baseline["trigger_columns"] == []
    assert baseline["replayed_ok"] == N and baseline["api_failed"] == 0


def test_surge_window_drifts_on_hour_and_distance_only_and_weather_is_context(demo):
    surge = demo[0]["windows"][1]
    assert surge["dataset_drift"] is True
    assert sorted(surge["trigger_columns"]) == ["distance_km", "hour"]  # not dow, never weather
    assert "temperature_2m" in surge["context_columns"]  # warming from Jan-Feb to April: reported, never a trigger
    assert not set(surge["context_columns"]) & set(surge["trigger_columns"])
    tests = surge["trigger_tests"]
    assert tests["hour"]["drifted"] and tests["distance_km"]["drifted"] and not tests["dow"]["drifted"]


def test_each_window_is_measured_on_its_own_requests(demo):
    _, _, api, log, _ = demo
    assert log.resets == 2 and log.fetches == ["baseline", "surge"]
    assert len(api.logged) == N  # reset before the surge window emptied the baseline requests


# ---- 3. the MAE flag only fires above the threshold --------------------------------------------------------------------

def test_mae_flag_fires_only_above_the_threshold():
    assert mae_flag(10.0, 11.0) is False  # exactly +10% is not "more than"
    assert mae_flag(10.0, 11.01) is True
    assert mae_flag(10.0, 9.0) is False and mae_flag(10.0, 10.0) is False
    assert mae_flag(10.0, 11.5, threshold=0.2) is False and mae_flag(10.0, 12.5, threshold=0.2) is True


def test_mae_comes_from_true_durations_and_the_synthetic_inflation(tmp_path, march_tail, weather, reference):
    april = make_trips(6_000, 20, "2026-04-01", 30, "2026-04", duration=20.0)  # every true duration is 20
    report, _, _, _ = run(tmp_path, april, march_tail, weather, reference, model=ConstantModel(10.0))
    base, surge = report["windows"]
    assert base["mae"] == pytest.approx(10.0)  # |20 - 10|
    assert surge["mae"] == pytest.approx(18.0)  # |28 - 10|: durations x1.4, labelled SYNTHETIC
    assert "SYNTHETIC" in surge["mae_source"] and surge["synthetic"] is True and base["synthetic"] is False
    assert report["flags"]["mae_over_threshold"] is True and report["flags"]["mae_change_vs_baseline"] == pytest.approx(0.8)
    quiet = run(tmp_path, april, march_tail, weather, reference, model=ConstantModel(10.0),
                cfg=demo_cfg(tmp_path, mae_threshold=0.9))[0]
    assert quiet["flags"]["mae_over_threshold"] is False  # +80% is not above a 90% threshold


def test_point_in_time_features_for_the_first_april_day_use_march_history():
    march = pd.DataFrame({"pickup_ts": pd.to_datetime(["2026-03-31 22:30", "2026-03-30 10:00"]),
                          "PULocationID": [5, 5], "DOLocationID": [6, 6], "duration_min": [20.0, 50.0],
                          "month": "2026-03"})
    april = pd.DataFrame({"pickup_ts": pd.to_datetime(["2026-04-01 00:30"]), "PULocationID": [5], "DOLocationID": [6],
                          "duration_min": [18.0], "month": "2026-04"})
    history = build_history(march, april)
    assert len(history) == 2  # the 3/31 trip is kept; the one from 3/30 10:00 is older than 27h and dropped
    with_march = rolling_pair_features(history, april)
    assert with_march["avg_duration_24h"].iloc[0] == pytest.approx(20.0)  # the March trip completed before 00:00
    assert with_march["trip_count_24h"].iloc[0] == 1
    cold = rolling_pair_features(april, april)  # what you would get without the March tail
    assert np.isnan(cold["avg_duration_24h"].iloc[0]) and cold["trip_count_24h"].iloc[0] == 0


def test_score_uses_the_march_tail_for_rolling_features(weather):
    class RollingEcho(ConstantModel):
        needs_rolling = True

        def predict(self, features):
            return features["avg_duration_24h"].fillna(-1000.0)

    march = pd.DataFrame({"pickup_ts": pd.to_datetime(["2026-03-31 22:30"]), "PULocationID": [5], "DOLocationID": [6],
                          "duration_min": [20.0], "month": "2026-03"})
    april = pd.DataFrame({"pickup_ts": pd.to_datetime(["2026-04-01 00:30"]), "PULocationID": [5], "DOLocationID": [6],
                          "duration_min": [20.0], "month": "2026-04"})
    assert score(RollingEcho(), april, weather, build_history(march, april))["mae"] == pytest.approx(0.0)
    assert score(RollingEcho(), april, weather, april)["mae"] == pytest.approx(1020.0)  # cold start without March


# ---- 4. retrain trigger and the real registry --------------------------------------------------------------------------

def test_retrain_runs_once_when_a_trigger_fires_and_reports_the_challenger(demo):
    report, calls, _, _, _tmp = demo
    assert len(calls) == 1 and Path(calls[0]).name == "drift_report.json" and Path(calls[0]).parent.name == "surge"
    surge = report["windows"][1]
    assert surge["retrain_triggered"] is True and surge["challenger_version"] == "2"
    assert report["windows"][0]["retrain_triggered"] is False


def test_no_retrain_flag_and_a_failing_retrain_do_not_break_the_summary(tmp_path, april, march_tail, weather, reference):
    report, calls, _, _ = run(tmp_path, april, march_tail, weather, reference, cfg=demo_cfg(tmp_path, retrain=False))
    assert calls == [] and report["retrain"]["triggered"] is False

    def broken(_path):
        raise RuntimeError("training exploded")

    report, _, _, _ = run(tmp_path, april, march_tail, weather, reference, retrain=broken)
    assert report["retrain"]["triggered"] is False and "training exploded" in report["retrain"]["reason"]
    assert "Retrain:" in format_summary(report)


def test_throwaway_store_guard_refuses_anything_but_a_sqlite_file_inside_the_work_dir(tmp_path, monkeypatch):
    work = tmp_path / "work"
    ok = f"sqlite:///{(work / 'retrain' / 'mlflow.db').as_posix()}"
    assert assert_throwaway_store(ok, work).name == "mlflow.db"
    for bad in ("http://localhost:5000", f"sqlite:///{(tmp_path / 'elsewhere.db').as_posix()}",
                f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"):
        with pytest.raises(DemoError):
            assert_throwaway_store(bad, work)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", ok)  # even a work-dir URI is refused when it is the configured real one
    with pytest.raises(DemoError, match="real tracking store"):
        assert_throwaway_store(ok, work)


def test_retrain_writes_only_inside_the_work_dir_and_never_touches_the_real_registry(tmp_path, monkeypatch):
    from mlflow import MlflowClient

    real_db = ROOT / "mlflow.db"
    before = (real_db.exists(), real_db.stat().st_mtime_ns if real_db.exists() else 0)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://real-registry.invalid:5000")
    forbidden = ("set_registered_model_alias", "create_registered_model", "create_model_version",
                 "delete_registered_model_alias", "transition_model_version_stage")
    for name in forbidden:
        monkeypatch.setattr(MlflowClient, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("registry write")))
    raw_src, holdout = tmp_path / "raw_src", tmp_path / "holdout" / "yellow_tripdata_2026-04.parquet"
    raw_src.mkdir()
    holdout.parent.mkdir()
    for name in ("yellow_tripdata_2026-01.parquet", "yellow_tripdata_2026-02.parquet", "yellow_tripdata_2026-03.parquet",
                 "taxi_zone_lookup.csv", "weather_2026_01_07.json"):
        (raw_src / name).write_text("x")
    holdout.write_text("x")
    seen = {}

    def fake_flow(**kwargs):
        seen.update(kwargs)
        return {"retrained": True, "reason": "new data", "version": "1", "mae": 3.8, "baseline_mae": 4.7}

    work = tmp_path / "work"
    out = retrain_challenger(tmp_path / "drift_report.json", work, 1000, holdout, raw_src, flow=fake_flow)
    assert out["triggered"] is True and out["challenger_version"] == "1" and out["train_rows"] == 1000
    assert seen["tracking_uri"].startswith("sqlite:///") and "real-registry" not in seen["tracking_uri"]
    for key in ("raw_dir", "processed_dir", "feast_root"):
        assert work.resolve() in Path(seen[key]).resolve().parents  # never data/raw, data/processed or data/feast
    assert seen["max_train_rows"] == 1000
    assert {p.name for p in Path(seen["raw_dir"]).iterdir()} >= {"yellow_tripdata_2026-04.parquet",
                                                                 "yellow_tripdata_2026-01.parquet"}
    assert (real_db.exists(), real_db.stat().st_mtime_ns if real_db.exists() else 0) == before


def test_a_full_retrain_trains_on_every_row(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("yellow_tripdata_2026-01.parquet", "yellow_tripdata_2026-02.parquet", "yellow_tripdata_2026-03.parquet",
                 "taxi_zone_lookup.csv", "weather_2026_01_07.json"):
        (raw / name).write_text("x")
    holdout = tmp_path / "yellow_tripdata_2026-04.parquet"
    holdout.write_text("x")
    seen = {}
    out = retrain_challenger(None, tmp_path / "work", None, holdout, raw,
                             flow=lambda **kw: seen.update(kw) or {"retrained": False, "reason": "no new data"})
    assert seen["max_train_rows"] is None and seen["drift_report"] is None and out["train_rows"] == "all"
    assert out["triggered"] is False


def test_the_demo_itself_never_registers_or_moves_an_alias(tmp_path, april, march_tail, weather, reference, monkeypatch):
    from mlflow import MlflowClient

    for name in ("set_registered_model_alias", "create_model_version", "create_registered_model", "create_run"):
        monkeypatch.setattr(MlflowClient, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("registry write")))
    report, _, _, _ = run(tmp_path, april, march_tail, weather, reference)
    assert report["retrain"]["triggered"] is True  # reached the (fake) retrain without any registry write


# ---- 5. API failures are reported, not raised --------------------------------------------------------------------------

def test_replay_counts_failures_and_stops_after_a_long_streak():
    trips = make_trips(100, 30, "2026-04-01", 5, "2026-04")
    ok = replay(trips, lambda payload: 200)
    assert (ok.sent, ok.ok, ok.failed, ok.aborted) == (100, 100, 0, False)
    down = replay(trips, lambda payload: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    assert down.aborted is True and down.sent == 25 and down.failed == 25 and "ConnectError" in down.errors[0]
    flaky = replay(trips, lambda payload: 500 if payload["pickup_zone"] % 2 else 200)
    assert flaky.failed + flaky.ok == 100 and flaky.errors and flaky.errors[0] == "HTTP 500"


def test_an_api_that_is_down_is_reported_and_the_summary_still_prints(tmp_path, april, march_tail, weather, reference):
    api = FakeApi(fail=lambda call: httpx.ConnectError("connection refused"))
    report, _calls, _, _ = run(tmp_path, april, march_tail, weather, reference, api=api)
    base, surge = report["windows"]
    for w in (base, surge):
        assert w["replayed_ok"] == 0 and w["replay_aborted"] is True and w["dataset_drift"] is None
        assert "no request succeeded" in w["drift_status"] and "ConnectError" in w["drift_status"]
        assert w["mae"] > 0  # the offline MAE does not depend on the API
    text = format_summary(report)
    assert "n/a" in text and "SYNTHETIC" in text
    assert (tmp_path / "report.json").exists()
    assert report["flags"]["dataset_drift"] is False  # an unmeasured drift is not a detected one


def test_http_errors_part_way_are_counted_and_the_drift_check_still_runs(tmp_path, april, march_tail, weather, reference):
    api = FakeApi(fail=lambda call: 503 if call % 4 == 0 else None)  # every 4th request fails
    report, _, _, _ = run(tmp_path, april, march_tail, weather, reference, api=api)
    base = report["windows"][0]
    assert base["api_failed"] == N // 4 and base["replayed_ok"] == N - N // 4
    assert base["dataset_drift"] is False and base["api_errors"][0] == "HTTP 503"
    assert "failed)" in format_summary(report)


def test_a_log_that_cannot_be_reset_is_reported_per_window(tmp_path, april, march_tail, weather, reference):
    api = FakeApi()
    log = FakeLog(api, weather, reset_error="`docker compose stop api` failed: no such service")
    report, _, _, _ = run(tmp_path, april, march_tail, weather, reference, api=api, log=log)
    for w in report["windows"]:
        assert w["dataset_drift"] is None and "log unavailable" in w["drift_status"] and w["mae"] > 0
    assert api.calls == 0  # nothing was replayed into a log that could not be reset


def test_an_empty_log_is_reported_with_a_hint(tmp_path, april, march_tail, weather, reference):
    class EmptyLog(FakeLog):
        def fetch(self, name):
            return pd.DataFrame(columns=COLUMNS)

    api = FakeApi()
    report, _, _, _ = run(tmp_path, april, march_tail, weather, reference, api=api, log=EmptyLog(api, weather))
    assert "ETA_PREDICTION_LOG" in report["windows"][0]["drift_status"]


# ---- 6. the report and the plain statement -----------------------------------------------------------------------------

def test_report_and_summary_say_what_the_demo_does_and_does_not_show(demo):
    report, _, _, _, tmp = demo
    on_disk = json.loads((tmp / "report.json").read_text())
    assert on_disk["windows"][1]["synthetic"] is True and "SYNTHETIC" in on_disk["synthetic"]
    for text in (on_disk["disclaimer"], format_summary(report)):
        assert "DETECTING" in text and "RETRAINING" in text and "does not show the model adapting" in text
        assert "SYNTHETIC" in text
    assert on_disk["mae_reference"]["champion_validation_mae_march"] == 3.79
    assert on_disk["mae_reference"]["april_full_month_mae"] > 0
    summary = format_summary(report)
    for header in ("window", "rows", "MAE source", "MAE", "dataset_drift", "trigger cols drifted", "context cols",
                   "retrain", "challenger"):
        assert header in summary.splitlines()[0]
    assert "MAE flag (> +10% over baseline)" in summary


# ---- 7. docker compose plumbing (all commands mocked) ------------------------------------------------------------------

class Recorder:
    def __init__(self, fail_on=None):
        self.commands, self.fail_on = [], fail_on

    def __call__(self, cmd, **kwargs):
        self.commands.append(cmd[2:])  # drop "docker compose"
        if self.fail_on and cmd[2] == self.fail_on:
            raise subprocess.CalledProcessError(1, cmd, stderr="boom")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def healthy(url, timeout):
    return httpx.Response(200)


def test_compose_log_reset_and_fetch_run_the_expected_commands(tmp_path):
    rec = Recorder()
    log = ComposeLog("http://localhost:8000", tmp_path, run=rec, get=healthy, sleep=lambda s: None)
    log.reset()
    assert [c[0] for c in rec.commands] == ["stop", "run", "start"] and "--no-deps" in rec.commands[1]
    rec.commands.clear()
    assert log.fetch("baseline").empty  # no file was produced by the fake docker
    assert rec.commands == [["stop", "api"], ["cp", "api:/app/log/predictions.sqlite",
                                              str(tmp_path / "baseline_prediction_log.sqlite")], ["start", "api"]]


def test_compose_log_restarts_the_api_when_a_step_fails_and_reports_it(tmp_path):
    rec = Recorder(fail_on="cp")
    log = ComposeLog("http://localhost:8000", tmp_path, run=rec, get=healthy, sleep=lambda s: None)
    with pytest.raises(LogError, match="docker compose cp"):
        log.fetch("surge")
    assert rec.commands[-1] == ["start", "api"]  # the stack is not left with the API stopped
    missing = ComposeLog("http://x", tmp_path, run=lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("docker")),
                         get=healthy, sleep=lambda s: None)
    with pytest.raises(LogError):
        missing.reset()


def test_compose_log_gives_up_when_the_api_never_becomes_healthy(tmp_path):
    log = ComposeLog("http://localhost:8000", tmp_path, run=Recorder(), sleep=lambda s: None, timeout=0.0,
                     get=lambda url, timeout: (_ for _ in ()).throw(httpx.ConnectError("down")))
    with pytest.raises(LogError, match="did not become healthy"):
        log.reset()


# ---- 8. command line ---------------------------------------------------------------------------------------------------

def test_main_reports_a_missing_holdout_file_instead_of_a_traceback(tmp_path, capsys):
    code = sd.main(["--holdout", str(tmp_path / "nope.parquet"), "--report", str(tmp_path / "r.json"),
                    "--work", str(tmp_path / "w")])
    assert code == 2 and "holdout file not found" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_the_default_holdout_lives_outside_data_raw():
    assert sd.HOLDOUT == ROOT / "data" / "holdout" / "yellow_tripdata_2026-04.parquet"
    assert "raw" not in sd.HOLDOUT.parts and os.fspath(sd.REPORT).endswith("surge_demo_report.json")
