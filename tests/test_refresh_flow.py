"""Monthly refresh flow (L4). Network, MLflow, Feast and DVC are always replaced; nothing here touches the internet."""
import json
from pathlib import Path

import pandas as pd
import pytest
from flows import month_state, refresh_flow
from flows.refresh_flow import (
    LOOKAHEAD,
    TRIP_URL,
    RefreshError,
    Steps,
    candidate_months,
    download_file,
    extend_weather,
    find_new_month,
    merge_weather,
    month_after,
    month_hours,
    retrain_and_judge,
    run_refresh,
)

from validation.clean import discover_months

VARS = ("temperature_2m", "precipitation", "snowfall", "wind_speed_10m")


def weather_json(start, end, value=1.0):
    hours = pd.date_range(start, end, freq="h")
    return {"latitude": 40.7, "hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours],
                                         **{v: [value] * len(hours) for v in VARS}}}


# ---- months and finding the next one ---------------------------------------------------------------------------

def test_month_arithmetic_crosses_the_year():
    assert month_after("2026-03") == "2026-04" and month_after("2026-12") == "2027-01"
    assert candidate_months(["2026-02", "2026-03"]) == ["2026-04", "2026-05", "2026-06"][:LOOKAHEAD]
    assert len(month_hours("2026-02")) == 28 * 24 and len(month_hours("2026-12")) == 31 * 24


def test_403_and_404_mean_not_published_yet():
    for code in (403, 404):
        month, unpublished = find_new_month(["2026-03"], lambda url, c=code: c)
        assert month is None and unpublished == ["2026-04", "2026-05", "2026-06"]


def test_earliest_published_month_wins_and_gaps_are_reported():
    seen = []

    def probe(url):
        seen.append(url)
        return 200 if "2026-05" in url or "2026-06" in url else 404

    assert find_new_month(["2026-03"], probe) == ("2026-05", ["2026-04"])
    assert seen == [TRIP_URL.format(month="2026-04"), TRIP_URL.format(month="2026-05")]  # stops at the first hit


@pytest.mark.parametrize("code", [500, 503, 301, 401])
def test_any_other_status_is_an_error_not_silence(code):
    with pytest.raises(RefreshError, match=str(code)):
        find_new_month(["2026-03"], lambda url: code)


# ---- weather ------------------------------------------------------------------------------------------------------

def test_weather_already_covering_the_month_is_not_fetched(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(weather_json("2026-01-01", "2026-07-31 23:00")))
    before = path.read_bytes()
    assert extend_weather(path, "2026-04", lambda url: pytest.fail("must not fetch")) is False
    assert path.read_bytes() == before


def test_weather_is_fetched_and_merged_when_the_month_is_missing(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(weather_json("2026-01-01", "2026-03-31 23:00", value=1.0)))
    urls = []

    def fetch(url):
        urls.append(url)
        return weather_json("2026-04-01", "2026-04-30 23:00", value=9.0)

    assert extend_weather(path, "2026-04", fetch) is True
    assert "start_date=2026-04-01&end_date=2026-04-30" in urls[0] and "America%2FNew_York" in urls[0]
    hourly = json.loads(path.read_text())["hourly"]
    assert hourly["time"] == sorted(hourly["time"]) and len(hourly["time"]) == len(set(hourly["time"]))
    assert hourly["temperature_2m"][0] == 1.0 and hourly["temperature_2m"][-1] == 9.0  # old rows kept, new appended
    assert extend_weather(path, "2026-04", lambda url: pytest.fail("covered now")) is False
    assert not list(tmp_path.glob("*.tmp"))


def test_overlapping_rows_keep_the_existing_values():
    existing = weather_json("2026-03-30", "2026-04-01 05:00", value=1.0)
    merged = merge_weather(existing, weather_json("2026-04-01", "2026-04-30 23:00", value=9.0), "2026-04")
    hourly = merged["hourly"]
    assert hourly["temperature_2m"][hourly["time"].index("2026-04-01T03:00")] == 1.0
    assert hourly["temperature_2m"][hourly["time"].index("2026-04-01T06:00")] == 9.0
    assert merged["latitude"] == 40.7


@pytest.mark.parametrize("break_it", ["short", "missing_var", "ragged"])
def test_a_bad_weather_response_is_rejected_and_the_file_untouched(tmp_path, break_it):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(weather_json("2026-01-01", "2026-03-31 23:00")))
    before = path.read_bytes()
    bad = weather_json("2026-04-01", "2026-04-30 23:00")
    if break_it == "short":
        bad = weather_json("2026-04-01", "2026-04-29 23:00")
    elif break_it == "missing_var":
        del bad["hourly"]["snowfall"]
    else:
        bad["hourly"]["precipitation"] = bad["hourly"]["precipitation"][:-1]
    with pytest.raises(RefreshError):
        extend_weather(path, "2026-04", lambda url: bad)
    assert path.read_bytes() == before


# ---- download (a file:// URL exercises the real code without a network) ------------------------------------------------

def test_download_is_atomic_and_leaves_no_part_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 3_000_000)
    dest = tmp_path / "raw" / "yellow_tripdata_2026-04.parquet"
    download_file(src.as_uri(), dest)
    assert dest.read_bytes() == src.read_bytes() and not list(dest.parent.glob("*.part"))


def test_failed_download_leaves_neither_file_nor_part(tmp_path):
    dest = tmp_path / "raw" / "yellow_tripdata_2026-04.parquet"
    with pytest.raises(RefreshError):
        download_file((tmp_path / "does-not-exist").as_uri(), dest)
    assert not dest.exists() and not list(dest.parent.glob("*.part"))


# ---- the run ------------------------------------------------------------------------------------------------------------

class Harness:
    """Fake steps that record the order they were called in and write a minimal trips_clean.parquet."""

    def __init__(self, tmp_path, published=("2026-04",), fail_at=None, champion=True):
        self.tmp, self.calls, self.fail_at = tmp_path, [], fail_at
        self.published, self.champion = set(published), champion
        self.raw, self.processed = tmp_path / "raw", tmp_path / "processed"
        self.state = tmp_path / "state.json"
        self.raw.mkdir()
        for month in ("2026-01", "2026-02", "2026-03"):
            (self.raw / f"yellow_tripdata_{month}.parquet").write_bytes(b"old")
        (self.raw / refresh_flow.WEATHER_FILE).write_text(json.dumps(weather_json("2026-01-01", "2026-07-31 23:00")))

    def step(self, name):
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} blew up")

    def steps(self):
        def probe(url):
            self.step("probe")
            return 200 if any(m in url for m in self.published) else 404

        def download(url, dest):
            self.step("download")
            dest.write_bytes(b"new")

        def clean(raw_dir, out_dir, months):
            self.step("clean")
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"month": months, "pickup_ts": pd.to_datetime([f"{m}-15" for m in months])}
                         ).to_parquet(out_dir / "trips_clean.parquet")
            return [{"month": m, "rows_in": 10} for m in months]

        def score(trips, month, weather_path, uri):
            self.step("score")
            return {"version": "2", "mae": 3.9} if self.champion else None

        def materialize(trips, root):
            self.step("materialize")
            return "2026-04-30 00:00:00"

        def retrain(trips, month, months, raw_dir, reports, uri, max_rows, champion):
            self.step("retrain")
            self.retrain_args = (month, months, champion)
            return {"challenger": {"version": "3", "mae": 3.8}, "champion": champion, "verdict": None, "promoted": False}

        def finish(month, state_path):
            self.step("finish")
            return month_state.finish_month(month, state_path, dvc=lambda args: self.calls.append("dvc " + args[0]))

        return Steps(probe=probe, download=download, fetch_json=lambda u: pytest.fail("weather covers April"),
                     clean=clean, score=score, materialize=materialize, retrain=retrain, finish=finish,
                     log=lambda msg: None)

    def run(self, **kwargs):
        return run_refresh(self.raw, self.processed, self.tmp / "feast", self.state, "uri", steps=self.steps(), **kwargs)


def test_a_new_month_goes_through_every_step_in_order_and_is_recorded_last(tmp_path):
    h = Harness(tmp_path)
    result = h.run()
    assert h.calls == ["probe", "download", "clean", "score", "materialize", "retrain", "finish",
                       "dvc add", "dvc repro", "dvc push"]
    assert result["refreshed"] and result["month"] == "2026-04"
    assert result["months"] == ["2026-01", "2026-02", "2026-03", "2026-04"]
    assert h.retrain_args[0] == "2026-04" and h.retrain_args[1][-1] == "2026-04"
    assert month_state.load_state(h.state) == ["2026-04"]
    assert json.loads((h.processed / "refresh_2026-04.json").read_text())["month"] == "2026-04"
    assert (h.raw / "yellow_tripdata_2026-04.parquet").read_bytes() == b"new"


def test_nothing_published_does_nothing(tmp_path):
    h = Harness(tmp_path, published=())
    result = h.run()
    assert result["refreshed"] is False and result["checked"] == ["2026-04", "2026-05", "2026-06"]
    assert set(h.calls) == {"probe"} and not h.state.exists()


def test_dry_run_only_names_the_month(tmp_path):
    h = Harness(tmp_path)
    result = h.run(dry_run=True)
    assert result["month"] == "2026-04" and result["refreshed"] is False
    assert h.calls == ["probe"] and not h.state.exists()


def test_next_run_continues_after_the_newest_processed_month(tmp_path):
    h = Harness(tmp_path, published=("2026-04", "2026-05"))
    assert h.run()["month"] == "2026-04"
    assert h.run()["month"] == "2026-05"  # April is in the state file now, so the probe starts at May
    assert month_state.load_state(h.state) == ["2026-04", "2026-05"]
    h.published = set()
    assert h.run()["refreshed"] is False


@pytest.mark.parametrize("failing", ["download", "clean", "score", "materialize", "retrain", "finish"])
def test_any_failing_step_leaves_the_state_untouched(tmp_path, failing):
    h = Harness(tmp_path, fail_at=failing)
    month_state.save_state(["2026-03"], h.state)
    before = h.state.read_bytes()
    with pytest.raises(RuntimeError, match=failing):
        h.run()
    assert h.state.read_bytes() == before
    assert "dvc add" not in h.calls or failing == "finish"  # DVC only ever runs inside the last step


def test_a_dvc_failure_after_everything_else_still_leaves_the_state_alone(tmp_path):
    h = Harness(tmp_path)
    steps = h.steps()

    def failing_finish(month, state_path):
        return month_state.finish_month(month, state_path, dvc=lambda args: (_ for _ in ()).throw(
            month_state.DvcError("push failed")))

    steps.finish = failing_finish
    with pytest.raises(month_state.DvcError):
        run_refresh(h.raw, h.processed, h.tmp / "feast", h.state, "uri", steps=steps)
    assert not h.state.exists()
    assert h.run()["refreshed"]  # a rerun just repeats the month


def test_missing_raw_file_for_a_processed_month_is_a_clear_error(tmp_path):
    h = Harness(tmp_path)
    (h.raw / "yellow_tripdata_2026-02.parquet").unlink()
    with pytest.raises(RefreshError, match="2026-02.*dvc pull"):
        h.run()
    assert not h.state.exists()


def test_no_champion_is_reported_not_fatal(tmp_path):
    h = Harness(tmp_path, champion=False)
    assert h.run()["champion"] is None


def test_default_steps_use_the_real_network_helpers_only_when_not_replaced():
    steps = Steps()
    assert steps.probe is refresh_flow.http_status and steps.finish is month_state.finish_month


# ---- retrain + verdict (MLflow, training and evaluation replaced) -----------------------------------------------------

@pytest.fixture
def judge(monkeypatch, tmp_path):
    import importlib

    import training

    compare = importlib.import_module("evaluation.compare")  # the package re-exports a function named compare

    (tmp_path / refresh_flow.WEATHER_FILE).write_text(json.dumps(weather_json("2026-01-01", "2026-07-31 23:00")))
    pd.DataFrame({"LocationID": [1, 2], "Borough": "M", "Zone": "Z", "service_zone": "Y"}).to_csv(
        tmp_path / "taxi_zone_lookup.csv", index=False)
    seen = {}

    class Model:
        def __init__(self, version):
            self.version = version

    def fake_train(*args, **kwargs):
        seen.update(kwargs)
        return {"run_id": "r", "version": "3", "metrics": {"mae": 3.8, "baseline_mae": 4.7}}

    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr(training, "load_model", lambda alias, uri: Model("3"))
    monkeypatch.setattr(compare, "evaluate",
                        lambda model, trips, weather, history=None: {"mae": 3.5, "rmse": 6.0, "p90_abs_error": 7.5})
    trips = pd.DataFrame({"month": ["2026-03", "2026-04"], "pickup_ts": pd.to_datetime(["2026-03-02", "2026-04-02"])})
    return trips, tmp_path, seen


def run_judge(judge, champion):
    trips, raw, seen = judge
    out = retrain_and_judge(trips, "2026-04", ["2026-02", "2026-03", "2026-04"], raw, [], "uri", None, champion)
    return out, seen


def test_challenger_is_trained_on_earlier_months_with_the_shipped_config_and_never_promoted(judge):
    out, seen = run_judge(judge, {"version": "2", "mae": 4.0, "rmse": 6.5, "p90_abs_error": 8.0})
    assert seen["train_months"] == ["2026-02", "2026-03"] and seen["valid_months"] == ["2026-04"]
    assert seen["use_geo"] is True and seen["params"] == {"objective": "l1"}
    assert out["verdict"]["would_promote"] is True and out["promoted"] is False  # 12.5% better MAE


def test_verdict_rejects_a_challenger_that_is_not_enough_better(judge):
    out, _ = run_judge(judge, {"version": "2", "mae": 3.51, "rmse": 6.5, "p90_abs_error": 7.5})
    assert out["verdict"]["would_promote"] is False and "MAE improvement" in out["verdict"]["reasons"][0]


def test_no_verdict_without_a_champion_or_when_it_is_the_same_version(judge):
    assert run_judge(judge, None)[0]["verdict"] is None
    assert run_judge(judge, {"version": "3", "mae": 3.5, "rmse": 6.0, "p90_abs_error": 7.5})[0]["verdict"] is None


# ---- the DVC stage must pick up refreshed months --------------------------------------------------------------------------

def test_clean_stage_default_covers_every_raw_month(tmp_path):
    for month in ("2026-04", "2026-01", "2026-03"):
        (tmp_path / f"yellow_tripdata_{month}.parquet").write_bytes(b"")
    (tmp_path / "taxi_zone_lookup.csv").write_text("")
    assert discover_months(tmp_path) == ["2026-01", "2026-03", "2026-04"]
    source = (Path(__file__).resolve().parents[1] / "src" / "validation" / "clean.py").read_text()
    assert 'default=["2026-01", "2026-02", "2026-03"]' not in source  # the fixed list would drop refreshed months
