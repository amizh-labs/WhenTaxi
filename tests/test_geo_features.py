import math

import numpy as np
import pandas as pd
import pytest

from features import FEATURE_COLUMNS, GEO_COLUMNS, ROLLING_COLUMNS, build_features, geo_features, load_centroids

JFK, TIMES_SQ, LGA, EWR = 132, 230, 138, 1


def test_distance_matches_haversine_and_is_symmetric():
    c = load_centroids()
    out = geo_features([JFK, TIMES_SQ], [TIMES_SQ, JFK])
    a, b = c.loc[JFK], c.loc[TIMES_SQ]
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b.lon - a.lon) / 2) ** 2
    assert out["distance_km"].iloc[0] == pytest.approx(2 * 6371.0088 * math.asin(math.sqrt(h)), rel=1e-4)
    assert out["distance_km"].iloc[0] == pytest.approx(out["distance_km"].iloc[1], rel=1e-5)
    assert 15 < out["distance_km"].iloc[0] < 25  # JFK to Midtown is about 20 km as the crow flies


def test_bearing_is_a_compass_direction():
    out = geo_features([JFK, TIMES_SQ], [TIMES_SQ, JFK])
    assert 270 < out["bearing_deg"].iloc[0] < 360  # JFK is south-east of Midtown, so JFK -> Midtown heads north-west
    assert out["bearing_deg"].iloc[0] == pytest.approx((out["bearing_deg"].iloc[1] + 180) % 360, abs=1.5)


def test_same_zone_airport_and_borough_pair():
    out = geo_features([TIMES_SQ, TIMES_SQ, JFK, 2], [TIMES_SQ, 4, 2, 3])
    assert out["same_zone"].tolist() == [1, 0, 0, 0]
    assert out["distance_km"].iloc[0] == 0 and out["bearing_deg"].iloc[0] == 0
    assert out["airport_flag"].tolist() == [0, 0, 1, 0]  # JFK counts for the pair
    assert out["borough_pair"].iloc[0] == 33  # Manhattan -> Manhattan (code 3 * 10 + 3)
    assert out["borough_pair"].iloc[2] == 44  # Queens -> Queens
    assert geo_features([LGA, EWR], [TIMES_SQ, TIMES_SQ])["airport_flag"].tolist() == [1, 1]


@pytest.mark.parametrize("unknown", [264, 265, 9999])
def test_zone_without_centroid_gives_nan_not_an_error(unknown):
    out = geo_features([unknown, TIMES_SQ], [TIMES_SQ, unknown])
    assert out[["distance_km", "bearing_deg"]].isna().all().all()
    assert out["same_zone"].tolist() == [0, 0] and out["airport_flag"].tolist() == [0, 0]
    assert out["borough_pair"].tolist() == [63, 36]  # Unknown borough code 6


def test_series_input_keeps_index_and_build_features_appends_geo_last():
    trips = pd.DataFrame({"pickup_ts": pd.to_datetime(["2026-01-05 08:00", "2026-01-05 09:30"]),
                          "PULocationID": [TIMES_SQ, 264], "DOLocationID": [JFK, 4]}, index=[10, 20])
    weather = pd.DataFrame({"weather_hour": pd.date_range("2026-01-05", periods=24, freq="h"),
                            **{c: 0.0 for c in ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]}})
    rolling = pd.DataFrame({c: [1.0, np.nan] for c in ROLLING_COLUMNS}, index=trips.index)
    out = build_features(trips, weather, None, rolling, geo=True)
    assert list(out.columns) == FEATURE_COLUMNS + ROLLING_COLUMNS + GEO_COLUMNS
    assert list(out.index) == [10, 20] and np.isnan(out.loc[20, "distance_km"])
    assert list(build_features(trips, weather, None, rolling).columns) == FEATURE_COLUMNS + ROLLING_COLUMNS  # default off
    # training/serving parity: the same function on the same ids gives the same values
    pd.testing.assert_frame_equal(out[GEO_COLUMNS], geo_features(trips["PULocationID"], trips["DOLocationID"]))
