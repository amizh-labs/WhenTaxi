import pandas as pd
import pytest

from features import FEATURE_COLUMNS, WEATHER_COLUMNS, build_features, load_weather

LEAKAGE = {"trip_distance", "fare_amount", "total_amount", "tip_amount"}


@pytest.fixture
def weather():
    hours = pd.date_range("2026-01-05 00:00", periods=48, freq="h")
    frame = pd.DataFrame({"weather_hour": hours})
    # hour index encoded in every weather column so a wrong join is obvious
    for col in WEATHER_COLUMNS:
        frame[col] = range(48)
    return frame


def trips(*timestamps, with_target=True):
    frame = pd.DataFrame(
        {
            "pickup_ts": pd.to_datetime(list(timestamps)),
            "PULocationID": [10] * len(timestamps),
            "DOLocationID": [20] * len(timestamps),
        }
    )
    if with_target:
        frame["duration_min"] = 12.5
    return frame


def test_time_features_and_weekend(weather):
    # 2026-01-05 is a Monday, 2026-01-10 a Saturday
    out = build_features(trips("2026-01-05 08:30", "2026-01-10 23:59"), weather.assign())
    assert out["hour"].tolist() == [8, 23]
    assert out["dow"].tolist() == [0, 5]
    assert out["is_weekend"].tolist() == [0, 1]


def test_weather_join_uses_pickup_hour_not_next(weather):
    out = build_features(trips("2026-01-05 03:59"), weather)
    assert out["temperature_2m"].iloc[0] == 3  # hour 03, not 04


def test_missing_weather_keeps_row(weather):
    out = build_features(trips("2026-02-01 00:00"), weather)
    assert len(out) == 1
    assert out[WEATHER_COLUMNS].isna().all(axis=None)


def test_no_leakage_columns(weather):
    extra = trips("2026-01-05 08:00").assign(trip_distance=3.0, fare_amount=9.0, total_amount=12.0)
    out = build_features(extra, weather)
    assert not LEAKAGE & set(out.columns)
    assert list(out.columns[: len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS


def test_serving_mode_without_target(weather):
    out = build_features(trips("2026-01-05 08:00", with_target=False), weather)
    assert "duration_min" not in out.columns


def test_training_mode_keeps_target_and_index(weather):
    frame = trips("2026-01-05 08:00", "2026-01-05 09:00")
    frame.index = [7, 3]
    out = build_features(frame, weather)
    assert out.index.tolist() == [7, 3]
    assert out["duration_min"].tolist() == [12.5, 12.5]


def test_load_weather_real_file_shape(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(
        '{"hourly": {"time": ["2026-01-01T00:00", "2026-01-01T01:00"],'
        ' "temperature_2m": [-1.0, 0.5], "precipitation": [0, 0.1],'
        ' "snowfall": [0, 0], "wind_speed_10m": [5, 6]}}'
    )
    frame = load_weather(path)
    assert list(frame.columns) == ["weather_hour", *WEATHER_COLUMNS]
    assert frame["weather_hour"].iloc[1] == pd.Timestamp("2026-01-01 01:00")
