"""Single feature-building function shared by training and serving.

Only request-time information is used: pickup zone, dropoff zone, pickup time and
weather at the pickup hour. Post-trip columns (distance, fares, tips) never enter.
"""
import pandas as pd

from features.geo import geo_features
from features.rolling import ROLLING_COLUMNS
from features.weather import WEATHER_COLUMNS
from features.zones import apply_zone_categories

ZONE_COLUMNS = ["PULocationID", "DOLocationID"]
TIME_COLUMNS = ["hour", "dow", "is_weekend"]
FEATURE_COLUMNS = ["PULocationID", "DOLocationID", *TIME_COLUMNS, *WEATHER_COLUMNS]
TARGET = "duration_min"


def build_features(
    trips: pd.DataFrame,
    weather: pd.DataFrame,
    zone_categories: list[int] | None = None,
    rolling: pd.DataFrame | None = None,
    geo: bool = False,
) -> pd.DataFrame:
    """Build model features.

    rolling: ROLLING_COLUMNS frame aligned to ``trips`` (features.rolling for training/evaluation,
    the Feast online store for serving). Appended after FEATURE_COLUMNS when given.

    geo: append GEO_COLUMNS (features.geo, static zone-pair geography; model trained with them needs them again).

    zone_categories: fixed zone list (features.zones). When given, the two zone columns are
    categorical with exactly these categories; unseen zones become NaN.

    trips: needs pickup_ts (naive NYC local time), PULocationID, DOLocationID;
           duration_min is passed through if present (training), absent when serving.
    weather: output of load_weather (weather_hour + WEATHER_COLUMNS).
    Row order and index of ``trips`` are preserved.
    """
    out = pd.DataFrame(index=trips.index)
    out["PULocationID"] = trips["PULocationID"].astype("int64")
    out["DOLocationID"] = trips["DOLocationID"].astype("int64")

    pickup = pd.to_datetime(trips["pickup_ts"])
    out["hour"] = pickup.dt.hour
    out["dow"] = pickup.dt.day_of_week
    out["is_weekend"] = (out["dow"] >= 5).astype("int64")

    # Point-in-time: weather of the hour containing the pickup, never a later hour.
    pickup_hour = pickup.dt.floor("h").rename("weather_hour")
    lookup = weather.drop_duplicates("weather_hour").set_index("weather_hour")
    joined = lookup.reindex(pickup_hour)[WEATHER_COLUMNS]
    joined.index = trips.index
    out = out.join(joined)

    out = out[FEATURE_COLUMNS]
    if rolling is not None:
        out = out.join(rolling[ROLLING_COLUMNS])
    if geo:
        out = out.join(geo_features(out["PULocationID"], out["DOLocationID"]))
    if zone_categories is not None:
        for col in ZONE_COLUMNS:
            out[col] = apply_zone_categories(out[col], zone_categories)
    if TARGET in trips.columns:
        out[TARGET] = trips[TARGET]
    return out
