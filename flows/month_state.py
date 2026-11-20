"""Processed-months state file + the DVC step that finishes a month.

A month counts as processed only after EVERYTHING succeeded, DVC included. ``finish_month`` therefore runs

    dvc add data/raw   ->   dvc repro   ->   dvc push

and writes the state file only after all three succeed. If any step fails the state file is left byte-for-byte as it
was (the next run simply repeats the month; all three commands are safe to repeat). The state file itself is written
atomically (temp file + rename), so a crash can never leave it half-written.

State file: data/processed/processed_months.json  ->  {"months": ["2026-04", ...], "updated_at": "<UTC>"}
"""
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "data" / "processed" / "processed_months.json"
DVC_STEPS: tuple[tuple[str, ...], ...] = (("add", "data/raw"), ("repro",), ("push",))


class DvcError(RuntimeError):
    """A DVC step failed; the state file was not touched."""


def load_state(path: Path = STATE_PATH) -> list[str]:
    """Processed months, sorted. A missing file means nothing processed yet; a corrupt one is an error, not a reset."""
    path = Path(path)
    if not path.exists():
        return []
    return sorted(json.loads(path.read_text())["months"])


def save_state(months: Sequence[str], path: Path = STATE_PATH) -> None:
    """Atomic write: readers see the old file or the new one, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"months": sorted(set(months)), "updated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run_dvc(args: Sequence[str], cwd: Path = ROOT) -> None:
    """``python -m dvc <args>`` (the dvc of the active environment). Raises DvcError with the output on failure."""
    cmd = [sys.executable, "-m", "dvc", *args]
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise DvcError(f"could not start `dvc {' '.join(args)}`: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip()[-500:]
        raise DvcError(f"`dvc {' '.join(args)}` failed (exit {result.returncode}): {tail}")


def finish_month(month: str, state_path: Path = STATE_PATH, dvc: Callable[[Sequence[str]], None] = run_dvc) -> list[str]:
    """Run the DVC steps for a fully successful ``month``, then (and only then) record it. Returns the new state."""
    months = load_state(state_path)  # read first: a corrupt state file stops us before any DVC work
    for step in DVC_STEPS:
        dvc(step)  # raises on failure -> the state file below is never reached
    if month not in months:
        months = sorted([*months, month])
    save_state(months, state_path)
    return months
