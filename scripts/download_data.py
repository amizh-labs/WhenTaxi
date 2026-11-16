"""Download raw data into data/raw (cross-platform, pathlib only).

Usage: python scripts/download_data.py [--months 2026-01 2026-02 2026-03]
"""
import argparse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude=40.7128&longitude=-74.006"
    "&start_date=2026-01-01&end_date=2026-07-31"
    "&hourly=temperature_2m,precipitation,snowfall,wind_speed_10m"
    "&timezone=America%2FNew_York"
)


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip (exists): {dest.name}")
        return
    print(f"downloading {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=["2026-01", "2026-02", "2026-03"])
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for month in args.months:
        fetch(TRIP_URL.format(month=month), RAW / f"yellow_tripdata_{month}.parquet")
    fetch(ZONES_URL, RAW / "taxi_zone_lookup.csv")
    fetch(WEATHER_URL, RAW / "weather_2026_01_07.json")


if __name__ == "__main__":
    main()
