"""Static zone-pair geography features, one shared function for training, evaluation and serving.

Runtime reads ONLY data/reference/zone_centroids.csv (built once by scripts/build_zone_centroids.py), so no
geospatial library is needed here. Everything is known at request time (zone ids), never from the trip itself.
A zone without a centroid (e.g. 264/265 'Unknown') yields NaN distance/bearing instead of an error.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ETA_CENTROIDS_PATH: where the CSV lives when the package is pip-installed (the Docker image), not run from the repo.
CENTROIDS_PATH = Path(os.environ.get("ETA_CENTROIDS_PATH")
                      or Path(__file__).resolve().parents[2] / "data" / "reference" / "zone_centroids.csv")
GEO_COLUMNS = ["distance_km", "bearing_deg", "same_zone", "borough_pair", "airport_flag"]
EARTH_RADIUS_KM = 6371.0088
BOROUGHS = ["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens", "Staten Island", "Unknown"]  # fixed code order
UNKNOWN_BOROUGH = BOROUGHS.index("Unknown")


@lru_cache(maxsize=4)
def load_centroids(path: str | Path = CENTROIDS_PATH) -> pd.DataFrame:
    """LocationID-indexed frame: lat, lon, borough_code, is_airport."""
    df = pd.read_csv(path).set_index("LocationID")
    df["borough_code"] = df["borough"].map({b: i for i, b in enumerate(BOROUGHS)}).fillna(UNKNOWN_BOROUGH).astype(int)
    df["is_airport"] = (df["service_zone"].isin(["Airports", "EWR"])).astype(int)
    return df


def geo_features(pu, do, centroids: pd.DataFrame | None = None) -> pd.DataFrame:
    """pu, do: array-likes of LocationIDs. Returns GEO_COLUMNS (index follows ``pu`` if it is a Series).

    distance_km: haversine between zone centroids; bearing_deg: initial compass bearing 0-360 (pickup -> dropoff);
    same_zone: 1 if pickup zone == dropoff zone; borough_pair: pickup_borough_code * 10 + dropoff_borough_code;
    airport_flag: 1 if either zone is an airport (JFK, LaGuardia, Newark).
    """
    c = load_centroids() if centroids is None else centroids
    index = pu.index if isinstance(pu, pd.Series) else None
    pu = pd.Series(np.asarray(pu), index=index).astype("int64")
    do = pd.Series(np.asarray(do), index=index).astype("int64")
    lat1, lon1 = np.radians(pu.map(c["lat"]).to_numpy(float)), np.radians(pu.map(c["lon"]).to_numpy(float))
    lat2, lon2 = np.radians(do.map(c["lat"]).to_numpy(float)), np.radians(do.map(c["lon"]).to_numpy(float))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    bearing = np.degrees(np.arctan2(np.sin(dlon) * np.cos(lat2),
                                    np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)))
    bearing = np.where(distance == 0, 0.0, bearing % 360.0)  # same point: bearing undefined, use 0 (NaN stays NaN)
    bpu = pu.map(c["borough_code"]).fillna(UNKNOWN_BOROUGH).astype(int)
    bdo = do.map(c["borough_code"]).fillna(UNKNOWN_BOROUGH).astype(int)
    apu, ado = pu.map(c["is_airport"]).fillna(0).astype(int), do.map(c["is_airport"]).fillna(0).astype(int)
    return pd.DataFrame({
        "distance_km": distance.astype("float32"),
        "bearing_deg": np.asarray(bearing, dtype="float32"),
        "same_zone": (pu == do).astype("int64").to_numpy(),
        "borough_pair": (bpu * 10 + bdo).to_numpy(),
        "airport_flag": (apu | ado).astype("int64").to_numpy(),
    }, index=index)
