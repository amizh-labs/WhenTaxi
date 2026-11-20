"""Append-only log of /predict calls in SQLite (WAL), written by a background thread.

Purpose: real requests can be compared with the training data (monitoring/prediction_drift.py).
Off unless ETA_PREDICTION_LOG=<path> is set, so the 5-replica cluster demo (many pods, one file) keeps only
Prometheus metrics. The log is for the single-instance Compose / live deployment.

Never slows down or breaks /predict: ``log()`` only does a non-blocking put on a bounded queue. If the queue is
full the record is dropped (counted); every write error is swallowed (counted) and the writer retries later.
Retention: after each batch the oldest rows beyond ``max_rows`` are deleted.
"""
import contextlib
import math
import os
import queue
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

COLUMNS = ["ts", "pickup_zone", "dropoff_zone", "pickup_ts", "temperature_2m", "precipitation", "snowfall",
           "wind_speed_10m", "weather_source", "eta_minutes", "model_version"]
SCHEMA = ("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, "
          "pickup_zone INTEGER, dropoff_zone INTEGER, pickup_ts TEXT, temperature_2m REAL, precipitation REAL, "
          "snowfall REAL, wind_speed_10m REAL, weather_source TEXT, eta_minutes REAL, model_version TEXT)")
DEFAULT_MAX_ROWS = 100_000
DEFAULT_QUEUE_SIZE = 1000
BATCH = 100
_STOP = object()


def _finite(value):
    return None if value is None or math.isnan(float(value)) else float(value)  # NaN -> NULL


class PredictionLog:
    def __init__(self, path: str | Path, max_rows: int = DEFAULT_MAX_ROWS, queue_size: int = DEFAULT_QUEUE_SIZE):
        self.path, self.max_rows = Path(path), max_rows
        self.dropped = 0  # records lost because the queue was full
        self.errors = 0  # failed writes (the batch is lost; the API is unaffected)
        self.written = 0
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="prediction-log", daemon=True)
        self._thread.start()

    # ---- called from /predict: must be instant and must never raise -----------------------------------------------
    def log(self, *, pickup_zone, dropoff_zone, pickup_ts, weather: dict, weather_source, eta_minutes,
            model_version) -> bool:
        try:
            row = (datetime.now(UTC).isoformat(timespec="seconds"), int(pickup_zone), int(dropoff_zone),
                   str(pickup_ts), _finite(weather.get("temperature_2m")), _finite(weather.get("precipitation")),
                   _finite(weather.get("snowfall")), _finite(weather.get("wind_speed_10m")), str(weather_source),
                   _finite(eta_minutes), str(model_version))
            self._queue.put_nowait(row)
            return True
        except queue.Full:
            self.dropped += 1
        except Exception:  # noqa: BLE001  logging problems never reach the caller
            self.errors += 1
        return False

    def flush(self, timeout: float = 10.0) -> None:
        """Block until everything queued so far is written (tests, shutdown)."""
        done = threading.Event()
        threading.Thread(target=lambda: (self._queue.join(), done.set()), daemon=True).start()
        done.wait(timeout)

    def close(self, timeout: float = 5.0) -> None:
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            return
        self._thread.join(timeout)

    # ---- background writer ----------------------------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(SCHEMA)
        conn.commit()
        return conn

    def _run(self) -> None:
        conn = None
        stop = False
        while not stop:
            item = self._queue.get()
            batch = [item]
            while len(batch) < BATCH:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            stop = _STOP in batch
            rows = [b for b in batch if b is not _STOP]
            try:
                if rows:
                    conn = conn or self._connect()
                    conn.executemany(f"INSERT INTO predictions ({', '.join(COLUMNS)}) "
                                     f"VALUES ({', '.join('?' * len(COLUMNS))})", rows)
                    conn.execute("DELETE FROM predictions WHERE id <= (SELECT MAX(id) FROM predictions) - ?",
                                 (self.max_rows,))
                    conn.commit()
                    self.written += len(rows)
            except Exception:  # noqa: BLE001  fail open: drop the batch, reconnect next time
                self.errors += len(rows)
                with contextlib.suppress(Exception):
                    conn and conn.close()
                conn = None
            finally:
                for _ in batch:
                    self._queue.task_done()
        with contextlib.suppress(Exception):
            conn and conn.close()


def default_prediction_log() -> PredictionLog | None:
    """ETA_PREDICTION_LOG=<sqlite path> enables the log (ETA_PREDICTION_LOG_MAX_ROWS sets retention)."""
    path = os.environ.get("ETA_PREDICTION_LOG")
    if not path:
        return None
    return PredictionLog(path, max_rows=int(os.environ.get("ETA_PREDICTION_LOG_MAX_ROWS", DEFAULT_MAX_ROWS)))


def read_log(path: str | Path):
    """The whole log as a DataFrame (empty frame with the log columns if the file or table does not exist)."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5.0)
    try:
        return pd.read_sql_query(f"SELECT {', '.join(COLUMNS)} FROM predictions ORDER BY id", conn)
    except Exception:  # noqa: BLE001  e.g. the table was never created
        return pd.DataFrame(columns=COLUMNS)
    finally:
        conn.close()
