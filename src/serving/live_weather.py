"""Live hourly weather for /predict from Open-Meteo's FORECAST endpoint (free, no key).

Same four ``hourly`` variables, NYC local time and default units (degC, mm, cm, km/h) as the archive file the model
was trained on. NOT the ``current`` block: it can aggregate precipitation differently from the hourly values.
One request returns yesterday + the next days of hourly rows; it is cached for the clock hour, so the API makes at
most one outbound call per hour. A failure is remembered for FAILURE_RETRY_SECONDS so an outage costs one 2 s
timeout, not one per request. Any failure or an uncovered hour returns None (the caller falls back).
"""
import threading
import time
from collections.abc import Callable
from datetime import datetime

import pandas as pd

from features import WEATHER_COLUMNS

URL = "https://api.open-meteo.com/v1/forecast"
LATITUDE, LONGITUDE = 40.738136, -74.04254  # the grid point of the archive file's coordinates
TIMEOUT_SECONDS = 2.0
FAILURE_RETRY_SECONDS = 60
PAST_DAYS, FORECAST_DAYS = 1, 3
PARAMS = {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": ",".join(WEATHER_COLUMNS),
          "timezone": "America/New_York", "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS}


def http_get_json(url: str, params: dict, timeout: float) -> dict:
    import httpx  # serving extra

    response = httpx.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def parse_hourly(payload: dict) -> pd.DataFrame:
    """Open-Meteo JSON -> frame of weather_hour + WEATHER_COLUMNS. Raises ValueError/KeyError if malformed."""
    hourly = payload["hourly"]
    frame = pd.DataFrame({"weather_hour": pd.to_datetime(hourly["time"]),
                          **{c: pd.to_numeric(pd.Series(hourly[c]), errors="raise") for c in WEATHER_COLUMNS}})
    if frame.empty:
        raise ValueError("no hourly rows")
    return frame


class LiveWeather:
    """lookup(pickup_ts) -> {column: float} for the pickup hour, or None (network off, malformed, hour not covered)."""

    def __init__(self, fetch: Callable[[str, dict, float], dict] | None = None,
                 clock: Callable[[], float] = time.time):
        # resolved at call time, so tests can block or replace the real HTTP function
        self._fetch = fetch or (lambda url, params, timeout: http_get_json(url, params, timeout))
        self._clock = clock
        self._lock = threading.Lock()
        self._frame: pd.DataFrame | None = None
        self._frame_hour: int | None = None  # clock hour the cached frame was fetched in
        self._failed_at: float | None = None

    def _current_frame(self) -> pd.DataFrame | None:
        now = self._clock()
        hour = int(now // 3600)
        with self._lock:
            if self._frame is not None and self._frame_hour == hour:
                return self._frame
            if self._failed_at is not None and now - self._failed_at < FAILURE_RETRY_SECONDS:
                return None
            try:
                self._frame = parse_hourly(self._fetch(URL, PARAMS, TIMEOUT_SECONDS))
                self._frame_hour, self._failed_at = hour, None
            except Exception:  # noqa: BLE001  any network/parse problem means "no live weather", never a crash
                self._failed_at = now
                return None
            return self._frame

    def lookup(self, pickup_ts: datetime | pd.Timestamp) -> dict[str, float] | None:
        hour = pd.Timestamp(pickup_ts).floor("h")
        # Only the forecast window is worth a call: older/far-future hours come from the archive file instead.
        now_local = pd.Timestamp(self._clock(), unit="s", tz="UTC").tz_convert("America/New_York").tz_localize(None)
        if not (now_local.floor("D") - pd.Timedelta(days=PAST_DAYS) <= hour
                < now_local.floor("D") + pd.Timedelta(days=FORECAST_DAYS)):
            return None
        frame = self._current_frame()
        if frame is None:
            return None
        row = frame[frame["weather_hour"] == hour]
        if row.empty or row[WEATHER_COLUMNS].iloc[0].isna().any():
            return None
        return {c: float(row[c].iloc[0]) for c in WEATHER_COLUMNS}
