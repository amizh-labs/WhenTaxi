"""The committed data/reference/zone_centroids.csv, and (only with the optional `geo` extra) the script that builds it."""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "reference" / "zone_centroids.csv"
LOOKUP = ROOT / "data" / "raw" / "taxi_zone_lookup.csv"
NO_POLYGON = {264, 265}  # 'Unknown' zones have no polygon in the TLC shapefile


def km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


@pytest.fixture(scope="module")
def centroids():
    return pd.read_csv(CSV).set_index("LocationID")


def test_jfk_within_about_a_km(centroids):  # zone 132 'JFK Airport' (centroid of the whole airport polygon)
    row = centroids.loc[132]
    assert km(row.lat, row.lon, 40.6413, -73.7781) < 1.2


def test_times_square_within_about_a_km(centroids):  # zone 230 'Times Sq/Theatre District'
    row = centroids.loc[230]
    assert km(row.lat, row.lon, 40.7580, -73.9855) < 1.2


def test_every_centroid_in_nyc_bounding_box(centroids):
    assert centroids["lat"].between(40.4, 41.0).all()
    assert centroids["lon"].between(-74.3, -73.6).all()


def test_covers_every_lookup_zone_that_has_a_polygon(centroids):
    lookup_ids = set(pd.read_csv(LOOKUP)["LocationID"])
    assert lookup_ids - set(centroids.index) == NO_POLYGON
    assert set(centroids.index) <= lookup_ids and centroids.index.is_unique


def test_csv_has_metadata_columns(centroids):
    assert {"borough", "service_zone"} <= set(centroids.columns)
    assert centroids.loc[132, "service_zone"] == "Airports" and centroids.loc[230, "borough"] == "Manhattan"


# ---- the one-time script: skipped unless `pip install -e ".[geo]"` --------------------------------------------

def test_script_centroid_math_handles_holes_multipart_and_shared_ids(tmp_path):
    shapefile = pytest.importorskip("shapefile")
    pytest.importorskip("pyproj")
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_zone_centroids as script

    x0, y0 = 1_000_000.0, 200_000.0  # inside EPSG:2263 (NYC), feet
    square = lambda x, y, s: [(x, y), (x, y + s), (x + s, y + s), (x + s, y), (x, y)]  # clockwise = outer ring
    hole = lambda x, y, s: [(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]  # counter-clockwise = hole
    shp = tmp_path / "z"
    with shapefile.Writer(str(shp), shapeType=shapefile.POLYGON) as w:
        w.field("LocationID", "N")
        w.poly([square(x0, y0, 100)])
        w.record(1)  # plain 100x100 square: centroid at (x0+50, y0+50)
        w.poly([square(x0, y0, 100), hole(x0, y0, 50)])  # square with a 50x50 hole in the lower-left corner
        w.record(2)
        w.poly([square(x0, y0, 100)])  # LocationID 3 split over two records (multi-row) ...
        w.record(3)
        w.poly([square(x0 + 200, y0, 100)])
        w.record(3)
    lookup = tmp_path / "lookup.csv"
    pd.DataFrame({"LocationID": [1, 2, 3, 264], "Borough": ["Queens"] * 3 + ["Unknown"],
                  "Zone": list("abcd"), "service_zone": ["Boro Zone"] * 3 + [None]}).to_csv(lookup, index=False)
    out = script.build(Path(f"{shp}.shp"), lookup).set_index("LocationID")
    assert list(out.index) == [1, 2, 3]  # 264 has no polygon, so it is absent

    from pyproj import Transformer

    back = Transformer.from_crs("EPSG:4326", script.SHAPE_CRS, always_xy=True)
    xy = {z: back.transform(r.lon, r.lat) for z, r in out.iterrows()}
    assert xy[1] == pytest.approx((x0 + 50, y0 + 50), abs=1)
    # hole removes the lower-left quarter: centroid shifts toward the upper-right by 50/6 ft on each axis
    assert xy[2] == pytest.approx((x0 + 50 + 50 / 6, y0 + 50 + 50 / 6), abs=1)
    assert xy[3] == pytest.approx((x0 + 150, y0 + 50), abs=1)  # merged rows: midpoint of the two squares
