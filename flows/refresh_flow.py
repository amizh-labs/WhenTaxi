"""Monthly refresh: pick up the next month of NYC taxi data and push it through the whole platform (Prefect).

Run once:   python -m flows.refresh_flow [--dry-run] [--max-train-rows N]
Daily 07:00 check (does nothing until TLC publishes a new month):   python -m flows.refresh_flow --serve

One run handles ONE month, in this order. Every step is repeatable, and the month is recorded as processed only at the
very end (flows.month_state.finish_month: dvc add -> repro -> push, then the atomic state write), so any failure just
means the next run starts the same month again.

    1. find    probe the TLC file for each of the next LOOKAHEAD months after the newest processed one; HTTP 403/404
               means "not published yet" (anything else is an error). The earliest published month wins.
    2. download  yellow_tripdata_<month>.parquet into data/raw (temp file + rename, skipped when already there)
    3. weather   extend data/raw/weather_2026_01_07.json from the Open-Meteo archive when it does not cover the month
    4. clean     validation.run over every processed month + the new one (per-rule drop report, trips_clean.parquet)
    5. score     the CHAMPION on the new month (this is the "how is the live model doing on fresh data" number)
    6. features  re-materialize the Feast online store from all cleaned trips (writes materialized.json)
    7. retrain   a CHALLENGER on every earlier month, validated on the new one (same rule as flows.training_flow)
    8. verdict   champion vs challenger on the new month, judged by evaluation.promote.decide. NOTHING is promoted here:
               the verdict is only reported, promotion stays a deliberate step (python -m evaluation.promote).
    9. finish    dvc add/repro/push, then the processed-months state file.

Report: data/processed/refresh_<month>.json (also returned). Nothing in this module touches git.
"""
import argparse
import calendar
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger

from features import WEATHER_COLUMNS, load_weather, load_zone_categories
from flows import month_state
from flows.training_flow import DEFAULT_MONTHS, SHIPPED_PARAMS

ROOT = Path(__file__).resolve().parents[1]
TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude=40.7128&longitude=-74.006"
    "&start_date={start}&end_date={end}&hourly=" + ",".join(WEATHER_COLUMNS) + "&timezone=America%2FNew_York"
)
WEATHER_FILE = "weather_2026_01_07.json"
NOT_PUBLISHED = frozenset({403, 404})  # what the TLC bucket answers for a month that does not exist yet
LOOKAHEAD = 3  # months after the newest processed one that are probed
TIMEOUT_SECONDS = 60
CRON = "0 7 * * *"


class RefreshError(RuntimeError):
    """A step failed; the processed-months state was not touched, so the next run retries the same month."""


# ---- months -------------------------------------------------------------------------------------------------------

def month_after(month: str) -> str:
    return str(pd.Period(month, "M") + 1)


def candidate_months(processed: list[str], lookahead: int = LOOKAHEAD) -> list[str]:
    """The months after the newest processed one that could be published next."""
    out, month = [], max(processed)
    for _ in range(lookahead):
        month = month_after(month)
        out.append(month)
    return out


def month_hours(month: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{month}-01")
    return pd.date_range(start, start + pd.offsets.MonthBegin(1), freq="h", inclusive="left")


# ---- network (always injected in tests) -----------------------------------------------------------------------------

def http_status(url: str) -> int:
    """HEAD request -> status code. HTTP errors are returned as codes; a network failure is a RefreshError."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=TIMEOUT_SECONDS) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RefreshError(f"could not reach {url}: {exc}") from exc


def download_file(url: str, dest: Path) -> None:
    """Download to a temp file, then rename: an interrupted download never leaves a half file under the real name."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        os.replace(tmp, dest)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RefreshError(f"download of {url} failed: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def fetch_json(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise RefreshError(f"request to {url} failed: {exc}") from exc


# ---- step 1: find the month -----------------------------------------------------------------------------------------

def find_new_month(processed: list[str], probe: Callable[[str], int] = http_status) -> tuple[str | None, list[str]]:
    """(earliest published month after the newest processed one | None, months probed that were not published)."""
    unpublished = []
    for month in candidate_months(processed):
        status = probe(TRIP_URL.format(month=month))
        if status == 200:
            return month, unpublished
        if status not in NOT_PUBLISHED:
            raise RefreshError(f"unexpected HTTP {status} probing {month}")
        unpublished.append(month)
    return None, unpublished


# ---- step 3: weather ------------------------------------------------------------------------------------------------

def merge_weather(existing: dict, new: dict, month: str) -> dict:
    """Existing hourly rows win; rows of ``new`` fill the gaps. ``new`` must be complete and well-formed."""
    hourly = new.get("hourly") or {}
    keys = ["time", *WEATHER_COLUMNS]
    if any(k not in hourly for k in keys) or len({len(hourly[k]) for k in keys}) != 1:
        raise RefreshError("weather response is missing a variable or has ragged columns")
    missing = set(month_hours(month).strftime("%Y-%m-%dT%H:%M")) - set(hourly["time"])
    if missing:
        raise RefreshError(f"weather response misses {len(missing)} hours of {month}")
    rows = {t: [hourly[c][i] for c in WEATHER_COLUMNS] for i, t in enumerate(hourly["time"])}
    old = existing["hourly"]
    for i, t in enumerate(old["time"]):
        rows[t] = [old[c][i] for c in WEATHER_COLUMNS]
    times = sorted(rows)
    merged = {"time": times, **{c: [rows[t][j] for t in times] for j, c in enumerate(WEATHER_COLUMNS)}}
    return {**existing, "hourly": merged}


def extend_weather(path: Path, month: str, fetch: Callable[[str], dict] = fetch_json) -> bool:
    """Make ``path`` cover every hour of ``month``. Returns True when it had to fetch and rewrite the file."""
    existing = json.loads(path.read_text())
    have = set(existing["hourly"]["time"])
    if all(t in have for t in month_hours(month).strftime("%Y-%m-%dT%H:%M")):
        return False
    last = calendar.monthrange(int(month[:4]), int(month[5:]))[1]
    merged = merge_weather(existing, fetch(WEATHER_URL.format(start=f"{month}-01", end=f"{month}-{last:02d}")), month)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(merged))
    os.replace(tmp, path)
    return True


# ---- steps 5-8: score, features, retrain, verdict ---------------------------------------------------------------------

def score_champion(trips: pd.DataFrame, month: str, weather_path: Path, tracking_uri: str) -> dict | None:
    """Champion metrics on ``month`` (history = every earlier month); None when no champion is registered yet."""
    import mlflow

    from evaluation.compare import evaluate
    from training import load_model

    try:
        champion = load_model("champion", tracking_uri)
    except mlflow.exceptions.MlflowException:
        return None
    new = trips[trips["month"] == month]
    return {"version": champion.version, **evaluate(champion, new, load_weather(weather_path), history=trips)}


def materialize_features(trips: pd.DataFrame, feast_root: Path) -> str:
    from features.store import build_online_store

    upto = trips["pickup_ts"].max().ceil("h")
    build_online_store(trips, feast_root, upto)
    return str(upto)


def retrain_and_judge(trips: pd.DataFrame, month: str, months: list[str], raw_dir: Path, reports: list[dict],
                      tracking_uri: str, max_train_rows: int | None, champion: dict | None) -> dict:
    """Train a challenger on every earlier month (validated on ``month``) and judge it against ``champion``."""
    from evaluation.compare import evaluate
    from evaluation.promote import decide
    from training import load_model, train

    result = train(trips, load_weather(raw_dir / WEATHER_FILE), load_zone_categories(raw_dir / "taxi_zone_lookup.csv"),
                   tracking_uri=tracking_uri, clean_report=reports, train_months=months[:-1], valid_months=[month],
                   max_train_rows=max_train_rows, params=SHIPPED_PARAMS, use_geo=True)
    trained = {"run_id": result["run_id"], "version": result["version"], "mae": result["metrics"]["mae"],
               "baseline_mae": result["metrics"]["baseline_mae"]}
    challenger = load_model("challenger", tracking_uri)
    chall = {"version": challenger.version,
             **evaluate(challenger, trips[trips["month"] == month], load_weather(raw_dir / WEATHER_FILE), history=trips)}
    out = {"challenger": chall, "trained": trained, "champion": champion, "verdict": None, "promoted": False}
    if champion is not None and champion["version"] != chall["version"]:
        ok, reasons = decide({"champion": champion, "challenger": chall})
        out["verdict"] = {"would_promote": ok, "reasons": reasons or ["passed all thresholds"]}
    return out


# ---- the run ----------------------------------------------------------------------------------------------------------

@dataclass
class Steps:
    """Everything with a side effect outside the files of this run; tests replace these."""
    probe: Callable = http_status
    download: Callable = download_file
    fetch_json: Callable = fetch_json
    clean: Callable | None = None  # default: validation.run
    score: Callable = score_champion
    materialize: Callable = materialize_features
    retrain: Callable = retrain_and_judge
    finish: Callable = month_state.finish_month
    log: Callable[[str], None] = field(default=lambda msg: print(msg, flush=True))


def run_refresh(raw_dir: Path = ROOT / "data" / "raw", processed_dir: Path = ROOT / "data" / "processed",
                feast_root: Path = ROOT / "data" / "feast", state_path: Path = month_state.STATE_PATH,
                tracking_uri: str | None = None, max_train_rows: int | None = None, dry_run: bool = False,
                steps: Steps | None = None) -> dict:
    from training import default_tracking_uri
    from validation import run as run_cleaning

    steps = steps or Steps()
    clean = steps.clean or run_cleaning
    tracking_uri = tracking_uri or default_tracking_uri()
    processed = month_state.load_state(state_path) or list(DEFAULT_MONTHS)  # empty state = the original 3 months

    month, unpublished = find_new_month(processed, steps.probe)
    if month is None:
        steps.log(f"no new month published (checked {', '.join(unpublished)})")
        return {"refreshed": False, "reason": "no new month published", "checked": unpublished, "processed": processed}
    steps.log(f"new month: {month}" + (f" (not yet published: {', '.join(unpublished)})" if unpublished else ""))
    if dry_run:
        return {"refreshed": False, "reason": "dry run", "month": month, "processed": processed}

    trip_file = raw_dir / f"yellow_tripdata_{month}.parquet"
    if not trip_file.exists():
        steps.download(TRIP_URL.format(month=month), trip_file)
    absent = [m for m in processed if not (raw_dir / f"yellow_tripdata_{m}.parquet").exists()]
    if absent:
        raise RefreshError(f"raw files for {', '.join(absent)} are missing; run `dvc pull` or scripts/download_data.py")
    weather_path = raw_dir / WEATHER_FILE
    weather_extended = extend_weather(weather_path, month, steps.fetch_json)

    months = sorted({*processed, month})
    reports = clean(raw_dir, processed_dir, months)
    trips = pd.read_parquet(processed_dir / "trips_clean.parquet")
    new_report = next(r for r in reports if r["month"] == month)

    champion = steps.score(trips, month, weather_path, tracking_uri)
    steps.log("champion on new month: " + ("none registered" if champion is None else f"MAE {champion['mae']:.3f}"))
    as_of = steps.materialize(trips, feast_root)
    judged = steps.retrain(trips, month, months, raw_dir, reports, tracking_uri, max_train_rows, champion)
    report = {"refreshed": True, "month": month, "months": months, "unpublished_before_it": unpublished,
              "weather_extended": weather_extended, "clean_report": new_report, "online_features_as_of": as_of, **judged}
    (processed_dir / f"refresh_{month}.json").write_text(json.dumps(report, indent=2, default=str))
    steps.finish(month, state_path)  # LAST: only now does the month count as processed
    return report


@flow(name="eta-monthly-refresh")
def refresh_flow(dry_run: bool = False, max_train_rows: int | None = None, tracking_uri: str | None = None) -> dict:
    log = get_run_logger()
    result = run_refresh(dry_run=dry_run, max_train_rows=max_train_rows, tracking_uri=tracking_uri,
                         steps=Steps(log=log.info))
    log.info("refresh: %s", result.get("reason") or f"{result['month']} processed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="only report which month would be processed")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--serve", action="store_true", help=f"run on a schedule (cron '{CRON}') instead of once")
    args = parser.parse_args()
    if args.serve:
        refresh_flow.serve(name="eta-monthly-refresh", cron=CRON,
                           parameters={"max_train_rows": args.max_train_rows})
    else:
        print(json.dumps(refresh_flow(args.dry_run, args.max_train_rows), indent=2, default=str))
