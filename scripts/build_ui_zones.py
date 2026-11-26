"""Build ui/zones.csv for the Streamlit demo UI.

Joins data/raw/taxi_zone_lookup.csv (names) with data/reference/zone_centroids.csv (lat/lon) and keeps only zones a
rider could pick: the Unknown and Outside-of-NYC zones (264, 265) have no centroid and are dropped, as is any other
zone without one. The UI never imports from src/ and reads only this small committed CSV.

Run:  python scripts/build_ui_zones.py
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["LocationID", "label", "Borough", "Zone", "lat", "lon"]


def build(lookup: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    zones = lookup.merge(centroids[["LocationID", "lat", "lon"]], on="LocationID", how="inner")
    zones = zones.dropna(subset=["Borough", "Zone", "lat", "lon"])
    zones = zones[~zones["Borough"].isin(["Unknown", "N/A"]) & ~zones["Zone"].isin(["Outside of NYC", "NV", "NA"])]
    zones = zones.assign(label=zones["Borough"] + " - " + zones["Zone"])
    dupes = zones["label"].duplicated(keep=False)  # a few zones share a name: add the id to keep labels unique
    zones.loc[dupes, "label"] = zones.loc[dupes, "label"] + " (" + zones.loc[dupes, "LocationID"].astype(str) + ")"
    return zones.sort_values(["Borough", "Zone", "LocationID"])[COLUMNS].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookup", type=Path, default=ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    parser.add_argument("--centroids", type=Path, default=ROOT / "data" / "reference" / "zone_centroids.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "ui" / "zones.csv")
    args = parser.parse_args()
    result = build(pd.read_csv(args.lookup), pd.read_csv(args.centroids))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, lineterminator="\n")
    print(f"wrote {len(result)} zones to {args.out}")
