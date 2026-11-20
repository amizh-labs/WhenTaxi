"""One-time build of data/reference/zone_centroids.csv from the TLC taxi-zone shapefile.

Source: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip  (NYC TLC "Taxi Zone Shapefile")
Downloaded: 2026-11-18. The shapefile is NOT committed; only the generated CSV is.

Needs the optional extra (not installed by default, not needed by CI, training or serving):
    pip install -e ".[geo]"
Run:
    python scripts/build_zone_centroids.py --download <empty dir>      # fetch + build
    python scripts/build_zone_centroids.py --shp <dir>/taxi_zones.shp   # build from an existing copy

Method: centroid per LocationID in the projected CRS of the shapefile (EPSG:2263, NY State Plane, US feet) as an
area-weighted polygon centroid over every ring of every record sharing that LocationID. The shapefile stores holes
with the opposite orientation of outer rings, so signed areas subtract them. The centroid is then converted to
WGS84 lat/lon with pyproj. Zones without a polygon (264, 265 'Unknown') are simply absent from the CSV.
"""
import argparse
import io
import itertools
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
SHAPE_CRS = "EPSG:2263"
COLUMNS = ["LocationID", "lat", "lon", "borough", "service_zone"]


def ring_area_centroid(points) -> tuple[float, float, float]:
    """Signed shoelace area and centroid of one ring (first point repeated at the end)."""
    a = cx = cy = 0.0
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a /= 2.0
    if a == 0.0:
        return 0.0, 0.0, 0.0
    return a, cx / (6.0 * a), cy / (6.0 * a)


def shape_moments(shape) -> tuple[float, float, float]:
    """(signed area, area*cx, area*cy) summed over all parts (outer rings and holes) of one polygon record."""
    pts = shape.points
    starts = [*shape.parts, len(pts)]
    total = sx = sy = 0.0
    for lo, hi in itertools.pairwise(starts):
        a, cx, cy = ring_area_centroid(pts[lo:hi])
        total, sx, sy = total + a, sx + a * cx, sy + a * cy
    return total, sx, sy


def build(shp: Path, lookup: Path) -> pd.DataFrame:
    import shapefile  # pyshp
    from pyproj import Transformer

    reader = shapefile.Reader(str(shp))
    fields = [f[0] for f in reader.fields[1:]]
    acc: dict[int, list[float]] = {}
    for rec, shape in zip(reader.records(), reader.shapes(), strict=True):
        loc = int(dict(zip(fields, rec, strict=True))["LocationID"])
        a, sx, sy = shape_moments(shape)
        m = acc.setdefault(loc, [0.0, 0.0, 0.0])
        m[0], m[1], m[2] = m[0] + a, m[1] + sx, m[2] + sy  # merge rows that share a LocationID
    to_wgs84 = Transformer.from_crs(SHAPE_CRS, "EPSG:4326", always_xy=True)
    rows = []
    for loc, (a, sx, sy) in sorted(acc.items()):
        if a == 0.0:
            continue
        lon, lat = to_wgs84.transform(sx / a, sy / a)
        rows.append({"LocationID": loc, "lat": round(lat, 6), "lon": round(lon, 6)})
    meta = pd.read_csv(lookup).rename(columns={"Borough": "borough"})[["LocationID", "borough", "service_zone"]]
    out = pd.DataFrame(rows).merge(meta, on="LocationID", how="left")
    out["borough"], out["service_zone"] = out["borough"].fillna("Unknown"), out["service_zone"].fillna("Unknown")
    return out[COLUMNS]


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(dest)
    return next(dest.rglob("*.shp"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shp", type=Path)
    p.add_argument("--download", type=Path, help="empty directory to download and extract the TLC zip into")
    p.add_argument("--lookup", type=Path, default=ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "reference" / "zone_centroids.csv")
    args = p.parse_args(argv)
    shp = args.shp or (download(args.download) if args.download else None)
    if shp is None:
        p.error("give --shp or --download")
    df = build(shp, args.lookup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} zone centroids to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
