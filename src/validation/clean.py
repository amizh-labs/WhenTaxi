"""Cleaning stage: raw TLC trips -> cleaned table (docs/contracts.md section 1) + drop report.

Rules are applied in order; each dropped row is attributed to the FIRST rule it breaks, so
the per-rule counts add up exactly: rows_in = sum(dropped) + rows_out.
"""
import json
from pathlib import Path

import pandas as pd

from validation.schema import CleanTripsSchema

RAW_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"]
MIN_DURATION_MIN = 1.0
MAX_DURATION_MIN = 180.0
UNKNOWN_ZONE_NAMES = {"unknown", "n/a", "outside of nyc"}


def load_zones(path: Path) -> tuple[set[int], set[int]]:
    """Return (all valid LocationIDs, IDs meaning unknown/outside NYC), read from the CSV."""
    zones = pd.read_csv(path)
    names = zones[["Borough", "Zone"]].apply(lambda col: col.astype("string").str.strip().str.lower())
    unknown = names.isin(UNKNOWN_ZONE_NAMES).any(axis=1)
    return set(zones["LocationID"]), set(zones.loc[unknown, "LocationID"])


def clean_trips(raw: pd.DataFrame, month: str, valid_zones: set[int], unknown_zones: set[int]):
    """Clean one month of raw trips. Returns (clean_df, report_dict)."""
    pickup = raw["tpep_pickup_datetime"]
    duration = (raw["tpep_dropoff_datetime"] - pickup).dt.total_seconds() / 60.0

    rules = {
        "dropped_outside_month": pickup.dt.strftime("%Y-%m") != month,
        "dropped_bad_duration": ~duration.between(MIN_DURATION_MIN, MAX_DURATION_MIN),
        "dropped_zone_not_in_lookup": ~(
            raw["PULocationID"].isin(valid_zones) & raw["DOLocationID"].isin(valid_zones)
        ),
        "dropped_unknown_or_outside_zone": (
            raw["PULocationID"].isin(unknown_zones) | raw["DOLocationID"].isin(unknown_zones)
        ),
    }

    keep = pd.Series(True, index=raw.index)
    report = {"month": month, "rows_in": len(raw)}
    for name, broken in rules.items():
        hit = keep & broken
        report[name] = int(hit.sum())
        keep &= ~broken
    report["rows_out"] = int(keep.sum())

    clean = pd.DataFrame(
        {
            "pickup_ts": pickup[keep],
            "PULocationID": raw.loc[keep, "PULocationID"].astype("int64"),
            "DOLocationID": raw.loc[keep, "DOLocationID"].astype("int64"),
            "duration_min": duration[keep],
            "month": month,
        }
    ).reset_index(drop=True)
    CleanTripsSchema.validate(clean)
    return clean, report


def clean_file(path: Path, zones_path: Path):
    """Clean a yellow_tripdata_YYYY-MM.parquet file; month comes from the file name."""
    month = path.stem.split("_")[-1]
    valid, unknown = load_zones(zones_path)
    raw = pd.read_parquet(path, columns=RAW_COLUMNS)
    return clean_trips(raw, month, valid, unknown)


def discover_months(raw_dir: Path) -> list[str]:
    """Months that have a raw trip file, sorted (YYYY-MM)."""
    return sorted(p.stem.split("_")[-1] for p in Path(raw_dir).glob("yellow_tripdata_*.parquet"))


def run(raw_dir: Path, out_dir: Path, months: list[str]) -> list[dict]:
    """Clean several months, write trips_clean.parquet and data_quality_report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, reports = [], []
    for month in months:
        clean, report = clean_file(raw_dir / f"yellow_tripdata_{month}.parquet", raw_dir / "taxi_zone_lookup.csv")
        frames.append(clean)
        reports.append(report)
        print(json.dumps(report))
    pd.concat(frames, ignore_index=True).to_parquet(out_dir / "trips_clean.parquet", index=False)
    (out_dir / "data_quality_report.json").write_text(json.dumps(reports, indent=2))
    return reports


if __name__ == "__main__":
    import argparse

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=None,
                        help="default: every yellow_tripdata_YYYY-MM.parquet in --raw-dir (so a refreshed month is "
                             "picked up by `dvc repro` instead of being silently dropped)")
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "processed")
    args = parser.parse_args()
    run(args.raw_dir, args.out_dir, args.months or discover_months(args.raw_dir))
