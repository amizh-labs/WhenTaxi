"""Stage L2: prediction log (SQLite WAL, fail-open) and drift on real requests."""
import json
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from monitoring.prediction_drift import (
    INSUFFICIENT,
    KS_ALPHA,
    LOG_DRIFT_COLUMNS,
    MIN_ROWS,
    TRIGGER_COLUMNS,
    check_log_drift,
    log_features,
    main,
    trigger_test,
)

from features import WEATHER_COLUMNS
from serving import create_app
from serving.prediction_log import COLUMNS, PredictionLog, default_prediction_log, read_log

LIVE = {"temperature_2m": 5.0, "precipitation": 0.3, "snowfall": 2.0, "wind_speed_10m": 11.0}


class StubModel:
    version = "7"
    zone_categories = tuple(range(1, 266))

    def predict(self, features):
        return features["snowfall"].fillna(0) + 10.0 + features["hour"]


class StubLive:
    def lookup(self, ts):
        return LIVE


def client_with_log(tmp_path, factory):
    return TestClient(create_app(StubModel, tmp_path / "none.json", rolling_provider=None,
                                 live_weather=lambda: StubLive(), prediction_log=factory))


def post(client, zone=161, ts="2026-11-18T14:30:00", **extra):
    return client.post("/predict", json={"pickup_zone": zone, "dropoff_zone": 237, "pickup_ts": ts, **extra})


# ---- the log ----------------------------------------------------------------------------------------------------

def test_every_predict_call_is_logged_with_all_fields_in_wal_mode(tmp_path):
    path = tmp_path / "log.sqlite"
    plog = PredictionLog(path)
    with client_with_log(tmp_path, lambda: plog) as client:
        assert post(client).status_code == 200
        assert post(client, zone=132, weather={"snowfall": 5.0}).status_code == 200
        plog.flush()
    rows = read_log(path)
    assert list(rows.columns) == COLUMNS and len(rows) == 2
    first, second = rows.iloc[0], rows.iloc[1]
    assert (first.pickup_zone, first.dropoff_zone, first.pickup_ts) == (161, 237, "2026-11-18T14:30:00")
    assert first[WEATHER_COLUMNS].tolist() == [5.0, 0.3, 2.0, 11.0] and first.weather_source == "live"
    assert first.eta_minutes == 2.0 + 10 + 14 and first.model_version == "7" and pd.notna(first.ts)
    assert second.weather_source == "request" and second.snowfall == 5.0
    assert sqlite3.connect(path).execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_log_is_off_by_default_and_enabled_by_env(tmp_path, monkeypatch):
    monkeypatch.delenv("ETA_PREDICTION_LOG", raising=False)
    assert default_prediction_log() is None
    with client_with_log(tmp_path, default_prediction_log) as client:  # off: no file, requests unaffected
        assert post(client).status_code == 200
    assert not list(tmp_path.glob("*.sqlite"))
    monkeypatch.setenv("ETA_PREDICTION_LOG", str(tmp_path / "on.sqlite"))
    monkeypatch.setenv("ETA_PREDICTION_LOG_MAX_ROWS", "7")
    plog = default_prediction_log()
    assert plog.max_rows == 7
    plog.close()


def test_unwritable_log_does_not_break_predict(tmp_path):
    plog = PredictionLog(tmp_path)  # a directory cannot be opened as a database
    with client_with_log(tmp_path, lambda: plog) as client:
        assert [post(client).status_code for _ in range(3)] == [200, 200, 200]
        plog.flush()
    assert plog.errors >= 1 and plog.written == 0


def test_a_log_that_raises_does_not_break_predict(tmp_path):
    class Exploding:
        def log(self, **kwargs):
            raise RuntimeError("disk on fire")

        def close(self):
            pass

    with client_with_log(tmp_path, Exploding) as client:
        resp = post(client)
    assert resp.status_code == 200 and resp.json()["eta_minutes"] == 26.0


def test_full_queue_drops_records_without_blocking(tmp_path, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(PredictionLog, "_connect", lambda self: release.wait() or (_ for _ in ()).throw(OSError("x")))
    plog = PredictionLog(tmp_path / "slow.sqlite", queue_size=2)
    start = time.perf_counter()
    results = [plog.log(pickup_zone=1, dropoff_zone=2, pickup_ts="t", weather={}, weather_source="x", eta_minutes=1.0,
                        model_version="1") for _ in range(50)]
    assert time.perf_counter() - start < 1.0  # never waited for the stuck writer
    assert results.count(False) >= 45 and plog.dropped >= 45
    release.set()
    plog.close()


def test_retention_keeps_only_the_newest_rows(tmp_path):
    path = tmp_path / "r.sqlite"
    plog = PredictionLog(path, max_rows=5)
    for i in range(20):
        plog.log(pickup_zone=1, dropoff_zone=2, pickup_ts="2026-01-01T00:00:00", weather={}, weather_source="archive",
                 eta_minutes=float(i), model_version="1")
    plog.flush()
    rows = read_log(path)
    assert len(rows) == 5 and rows["eta_minutes"].tolist() == [15.0, 16.0, 17.0, 18.0, 19.0]
    assert rows["temperature_2m"].isna().all()  # NaN/absent weather is stored as NULL
    plog.close()


# ---- drift on the logged window ---------------------------------------------------------------------------------

def make_log(n, seed, zones=None, snowfall=0.01, start="2026-01-05", hours=None, missing_weather=0.0):
    rng = np.random.default_rng(seed)
    pool = np.arange(1, 264) if zones is None else np.asarray(zones)
    minutes = rng.integers(0, 55 * 24 * 60, n)
    if hours is not None:  # keep the day, restrict the hour of day
        minutes = (minutes // 1440) * 1440 + rng.choice(hours, n) * 60 + rng.integers(0, 60, n)
    pickup = pd.Timestamp(start) + pd.to_timedelta(minutes, unit="m")
    log = pd.DataFrame({
        "ts": pd.date_range("2026-11-18 09:00", periods=n, freq="min").astype(str),
        "pickup_zone": rng.choice(pool, n), "dropoff_zone": rng.choice(pool, n),
        "pickup_ts": pickup.strftime("%Y-%m-%dT%H:%M:%S"),
        "temperature_2m": rng.normal(2, 5, n), "precipitation": rng.exponential(0.1, n),
        "snowfall": rng.exponential(snowfall, n), "wind_speed_10m": rng.normal(15, 5, n),
        "weather_source": "live", "eta_minutes": 15.0, "model_version": "3"})
    if missing_weather:
        log.loc[rng.random(n) < missing_weather, WEATHER_COLUMNS] = np.nan
    return log


@pytest.fixture(scope="module")
def reference_log():
    return make_log(5000, seed=0)


@pytest.fixture(scope="module")
def reference(reference_log):
    return log_features(reference_log)


def test_window_drawn_from_the_reference_shows_no_drift(reference_log, reference):
    window = reference_log.sample(600, random_state=1)
    result = check_log_drift(reference, window)
    assert result["dataset_drift"] is False and result["status"] == "ok" and result["n_rows"] == 600


def test_skewed_batch_heavy_snow_and_only_airports_shows_drift(reference, tmp_path):
    skewed = make_log(400, seed=2, zones=[1, 132, 138], snowfall=3.0)
    result = check_log_drift(reference, skewed, html_path=tmp_path / "r.html")
    assert result["dataset_drift"] is True
    assert result["drifted_columns"] == ["distance_km"]  # the only trigger column that moved
    assert "snowfall" in result["context_drift"]["drifted_columns"]  # reported as context
    assert (tmp_path / "r.html").exists()


# ---- single factors: each moves only its own column --------------------------------------------------------------

def test_snowfall_only_is_context_and_never_triggers(reference):
    result = check_log_drift(reference, make_log(500, seed=11, snowfall=3.0))
    assert result["context_drift"]["drifted_columns"] == ["snowfall"]
    assert result["drifted_columns"] == [] and result["dataset_drift"] is False


def test_airport_zones_only_flags_distance_and_no_other_trigger_column(reference):
    result = check_log_drift(reference, make_log(500, seed=12, zones=[1, 132, 138]))
    assert result["drifted_columns"] == ["distance_km"] and result["dataset_drift"] is True
    assert result["context_drift"]["drifted_columns"] == []


def test_evening_hours_only_flags_hour_and_no_other_trigger_column(reference):
    result = check_log_drift(reference, make_log(500, seed=13, hours=[18, 19, 20, 21, 22, 23]))
    assert result["drifted_columns"] == ["hour"] and result["dataset_drift"] is True
    assert result["context_drift"]["drifted_columns"] == []


def test_weather_never_sets_dataset_drift_even_when_every_weather_column_moves(reference_log, reference):
    window = reference_log.sample(500, random_state=21).assign(
        temperature_2m=lambda d: d.temperature_2m + 25, precipitation=5.0, snowfall=4.0, wind_speed_10m=60.0)
    result = check_log_drift(reference, window)
    assert set(result["context_drift"]["drifted_columns"]) == set(WEATHER_COLUMNS)
    assert result["dataset_drift"] is False and result["drifted_columns"] == [] and result["share_drifted"] == 0.0


# ---- effect-size floor and binned tests ---------------------------------------------------------------------------

def test_a_big_window_with_a_significant_but_tiny_shift_is_not_drift(reference):
    """20,000 requests make a 6% evening skew statistically significant, but it is far too small to matter."""
    window = pd.concat([make_log(18_800, seed=31), make_log(1_200, seed=32, hours=[18, 19, 20, 21, 22, 23])],
                       ignore_index=True)
    result = check_log_drift(reference, window)
    hour = result["trigger_tests"]["hour"]
    assert hour["p_value"] < KS_ALPHA  # a p-value alone would have flagged it
    assert hour["effect"] < hour["effect_floor"] and hour["drifted"] is False
    assert result["drifted_columns"] == [] and result["dataset_drift"] is False


def test_the_report_lists_the_test_behind_each_trigger_column(reference, reference_log):
    result = check_log_drift(reference, reference_log.sample(300, random_state=5))
    assert set(result["trigger_tests"]) == {"hour", "dow", "distance_km"}
    assert all({"p_value", "effect", "effect_floor", "drifted"} <= set(t) for t in result["trigger_tests"].values())


def test_trigger_test_on_identical_and_disjoint_hours():
    hours = pd.Series(np.tile(np.arange(24), 50), dtype=float)
    same = trigger_test("hour", hours, hours.sample(frac=1, random_state=0))
    assert same["effect"] == 0.0 and same["drifted"] is False
    night = pd.Series(np.tile(np.arange(0, 3), 400), dtype=float)  # hours 0-2 only: a different 3-hour block entirely
    moved = trigger_test("hour", hours, night)
    assert moved["effect"] > 0.8 and moved["drifted"] is True


def test_weekday_shift_is_caught_by_the_binned_dow_test():
    ref = pd.Series(np.tile(np.arange(7), 200), dtype=float)
    weekend = pd.Series(np.tile([5, 6], 300), dtype=float)
    assert trigger_test("dow", ref, weekend)["drifted"] is True
    assert trigger_test("dow", ref, ref.sample(frac=1, random_state=1))["drifted"] is False


def test_trigger_test_handles_one_category_and_empty_input():
    one = pd.Series([5.0] * 100)
    assert trigger_test("hour", one, one) == {"p_value": 1.0, "effect": 0.0, "effect_floor": 0.1, "drifted": False}
    assert trigger_test("dow", one, pd.Series([], dtype=float))["drifted"] is False


def test_distance_needs_both_a_significant_and_a_large_ks_difference():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(5, 2, 20_000))
    assert trigger_test("distance_km", ref, pd.Series(rng.normal(5.05, 2, 20_000)))["drifted"] is False  # tiny shift
    assert trigger_test("distance_km", ref, pd.Series(rng.normal(8, 2, 500)))["drifted"] is True  # big shift


# ---- false positives ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [200, 500, 2000])
def test_false_positive_rate_below_5_percent(n, reference_log, reference):
    """50 random windows of n requests drawn from the reference: how often does dataset_drift fire?"""
    fired = [check_log_drift(reference, reference_log.sample(n, random_state=i))["dataset_drift"] for i in range(50)]
    rate = sum(fired) / len(fired)
    print(f"false-positive rate, 50 windows of {n} rows: {rate:.1%} ({sum(fired)}/50)")
    assert rate < 0.05, f"dataset_drift fired in {sum(fired)}/50 windows of {n} rows"


# ---- report format, thin data, data quality ---------------------------------------------------------------------

def test_report_has_the_contract_keys_and_columns_exclude_zone_ids(reference, reference_log):
    result = check_log_drift(reference, reference_log.sample(300, random_state=3))
    assert {"month", "dataset_drift", "drifted_columns", "share_drifted"} <= set(result)  # docs/contracts.md section 5
    assert {"context_drift", "data_quality"} <= set(result)  # section 8
    assert {"share_drifted_features", "drifted_features"} <= set(result)  # what flows/retrain.py reads
    assert set(result["context_drift"]["columns"]) == set(WEATHER_COLUMNS)
    json.dumps(result)
    assert not {"PULocationID", "DOLocationID", "pickup_zone", "dropoff_zone"} & set(LOG_DRIFT_COLUMNS)
    assert set(TRIGGER_COLUMNS) == {"hour", "dow", "distance_km"}


def test_insufficient_data_below_the_minimum_never_flags_drift(reference):
    skewed = make_log(MIN_ROWS - 1, seed=4, zones=[132], snowfall=5.0)
    result = check_log_drift(reference, skewed)
    assert result["status"] == INSUFFICIENT and result["dataset_drift"] is False and result["n_rows"] == MIN_ROWS - 1
    assert check_log_drift(reference, make_log(MIN_ROWS, seed=4))["status"] == "ok"  # exactly the minimum is enough


def test_empty_or_missing_log_is_handled(reference, tmp_path):
    empty = read_log(tmp_path / "does_not_exist.sqlite")
    assert empty.empty and list(empty.columns) == COLUMNS
    assert check_log_drift(reference, empty)["status"] == INSUFFICIENT
    created = tmp_path / "empty.sqlite"
    PredictionLog(created).close()  # opened but nothing logged: no table yet
    assert read_log(created).empty
    out = tmp_path / "reports"
    assert main(["--log", str(tmp_path / "does_not_exist.sqlite"), "--out", str(out)]) == 0
    report = json.loads((out / "drift_report.json").read_text())
    assert report["status"] == INSUFFICIENT and report["dataset_drift"] is False and report["n_rows"] == 0
    assert report["data_quality"] == {"share_missing_weather": 0.0, "alert": False}


def test_missing_weather_is_data_quality_not_drift(reference, reference_log):
    window = reference_log.sample(400, random_state=5).assign(temperature_2m=np.nan, precipitation=np.nan,
                                                              snowfall=np.nan, wind_speed_10m=np.nan)
    result = check_log_drift(reference, window)
    assert result["data_quality"] == {"share_missing_weather": 1.0, "alert": True}
    assert result["dataset_drift"] is False and result["drifted_columns"] == []
    assert result["context_drift"]["columns"] == []  # nothing to compare, and nothing marked as drifted


def test_data_quality_alert_threshold_is_20_percent(reference):
    just_under = check_log_drift(reference, make_log(500, seed=6, missing_weather=0.10))
    over = check_log_drift(reference, make_log(500, seed=6, missing_weather=0.40))
    assert just_under["data_quality"]["alert"] is False
    assert 0.05 < just_under["data_quality"]["share_missing_weather"] < 0.2
    assert over["data_quality"]["alert"] is True and over["data_quality"]["share_missing_weather"] > 0.3
    assert over["dataset_drift"] is False  # partially missing weather never triggers anything by itself
