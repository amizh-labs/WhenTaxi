"""Manual sanity check: ONE real Open-Meteo forecast call, printed with units, for the current NYC hour.

    python scripts/check_live_weather.py

Compare the numbers with a weather app for NYC. Units must match the archive the model was trained on:
temperature_2m degC, precipitation mm, snowfall cm, wind_speed_10m km/h. Not part of the test suite (it uses the network).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import pandas as pd
import prometheus_client  # noqa: F401  (before feast-importing modules)

from features import WEATHER_COLUMNS
from serving.live_weather import PARAMS, URL, http_get_json, parse_hourly

UNITS_EXPECTED = {"temperature_2m": "°C", "precipitation": "mm", "snowfall": "cm", "wind_speed_10m": "km/h"}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # the degree sign on Windows consoles
    payload = http_get_json(URL, PARAMS, 5.0)
    units = payload.get("hourly_units", {})
    frame = parse_hourly(payload)
    now = pd.Timestamp.now(tz="America/New_York").tz_localize(None).floor("h")
    print(f"GET {URL} {json.dumps(PARAMS)}")
    print(f"rows: {len(frame)} ({frame['weather_hour'].min()} .. {frame['weather_hour'].max()}, NYC local time)")
    for label, hour in (("previous hour", now - pd.Timedelta(hours=1)), ("current hour", now), ("next hour", now + pd.Timedelta(hours=1))):
        row = frame[frame["weather_hour"] == hour]
        if row.empty:
            print(f"{label} {hour}: not in response")
            continue
        values = ", ".join(f"{c}={row[c].iloc[0]} {units.get(c, '?')}" for c in WEATHER_COLUMNS)
        print(f"{label} {hour}: {values}")
    bad = {c: units.get(c) for c in WEATHER_COLUMNS if units.get(c) != UNITS_EXPECTED[c]}
    print("UNITS OK (match the archive file)" if not bad else f"UNITS MISMATCH vs archive: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
