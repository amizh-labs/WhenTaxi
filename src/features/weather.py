"""Load the Open-Meteo hourly weather JSON into a frame keyed by local hour."""
import json
from pathlib import Path

import pandas as pd

WEATHER_COLUMNS = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"]


def load_weather(path: str | Path) -> pd.DataFrame:
    """Return columns: weather_hour (naive NYC local time) + WEATHER_COLUMNS."""
    hourly = json.loads(Path(path).read_text())["hourly"]
    frame = pd.DataFrame(hourly)
    frame["weather_hour"] = pd.to_datetime(frame["time"])
    return frame[["weather_hour", *WEATHER_COLUMNS]]
