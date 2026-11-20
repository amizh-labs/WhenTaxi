"""When was the Feast online store last materialized? A tiny JSON file next to the store, no Feast import.

features.store.materialize writes it after every successful materialization; the API turns it into the
``eta_feature_age_seconds`` gauge. Missing or unreadable file -> age is None (the gauge exports MISSING_AGE).
"""
import json
import time
from datetime import UTC, datetime
from pathlib import Path

FILENAME = "materialized.json"
MISSING_AGE = -1.0  # exported when the file is missing/unreadable: "unknown", never matches an age > threshold alert


def metadata_path(feast_root: str | Path) -> Path:
    return Path(feast_root) / FILENAME


def write_materialization_metadata(feast_root: str | Path, upto=None, now: float | None = None) -> Path:
    """Record 'materialized just now' (wall-clock) and the data cut-off ``upto`` the store was built as of."""
    path = metadata_path(feast_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(time.time() if now is None else now, tz=UTC)
    path.write_text(json.dumps({"materialized_at": stamp.isoformat(timespec="seconds"),
                                "upto": None if upto is None else str(upto)}, indent=2))
    return path


def feature_age_seconds(feast_root: str | Path, now: float | None = None) -> float | None:
    """Seconds since the last materialization, or None if unknown. Never raises."""
    try:
        data = json.loads(metadata_path(feast_root).read_text())
        written = datetime.fromisoformat(data["materialized_at"]).timestamp()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return max(0.0, (time.time() if now is None else now) - written)  # a clock a bit behind is "just now"
