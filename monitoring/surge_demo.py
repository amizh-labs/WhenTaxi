"""Evening-rush surge demo: the platform DETECTS a shift in traffic and RETRAINS a challenger.

    python -m monitoring.surge_demo [--url http://localhost:8000] [--n 2000] [--seed 0]
                                    [--holdout data/holdout/yellow_tripdata_2026-04.parquet]
                                    [--retrain-rows 300000 | --full-retrain] [--no-retrain]

WHAT THIS IS NOT: it does not show the model adapting to anything. The challenger is trained on Jan-Mar and validated
on April; it has never seen the surge. What the demo shows is the platform noticing a shift and reacting to it.

It runs against the Docker Compose stack (the API with the prediction log on, see docker-compose.yml) and uses cleaned
April 2026 trips, held out of training, as the replay source. Two windows are replayed through /predict:

  baseline  N April trips as they are.
  surge     N April trips resampled toward evening hours (16-21) with a higher airport share.

The check is split in two, on purpose:

  DRIFT is measured on what the API logged. It only needs pickup_ts and the two zones, so it is sound for replayed
  history. (/predict reads its rolling features, avg_duration_1h / avg_duration_24h / trip_count_24h, from ONE
  materialized snapshot and ignores pickup_ts, so replayed predictions are not point-in-time: see README, Limitations.)
  MAE is computed OFFLINE with the same code as the monthly evaluation (evaluation.compare.evaluate), with rolling
  features computed point-in-time over the last 27 hours of cleaned March plus April. The champion is only READ.
  The surge window's true durations are inflated by 40 percent. That is SYNTHETIC: a made-up ground truth, not data.

When drift or the MAE flag fires, the existing retrain flow (flows.training_flow.retrain_flow) runs with a throwaway
MLflow SQLite store, throwaway processed and Feast folders and a copy of the raw files, all under
data/processed/surge_demo/. The real registry, the real champion alias, data/processed and data/feast are never written.
Output: a summary table and data/processed/surge_demo_report.json.
"""
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from evaluation.compare import evaluate
from features import load_weather
from monitoring.prediction_drift import MIN_ROWS, check_log_drift, reference_features
from serving.prediction_log import read_log
from validation.clean import clean_file

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "data" / "holdout" / "yellow_tripdata_2026-04.parquet"
WORK = ROOT / "data" / "processed" / "surge_demo"
REPORT = ROOT / "data" / "processed" / "surge_demo_report.json"

AIRPORT_ZONES = (1, 132, 138)  # Newark, JFK, LaGuardia
EVENING_HOURS = tuple(range(16, 22))  # 16:00-21:59
EVENING_SHARE = 0.85  # share of the surge window picked up in the evening hours
AIRPORT_SHARE = 0.30  # share of trips to or from an airport (the baseline month has a few percent)
SYNTHETIC_FACTOR = 1.4  # SYNTHETIC: true durations x1.4 in the surge window's ground truth
MAE_FLAG_THRESHOLD = 0.10  # this demo's own threshold (not the CI gate's 2% or the promotion rules)
DEFAULT_N = 2000
HISTORY_HOURS = 24 + 3  # 24h window + the longest kept trip (3h), as in features.store.materialize
DEFAULT_REFERENCE_ROWS = 200_000
DEFAULT_RETRAIN_ROWS = 300_000
MAX_CONSECUTIVE_FAILURES = 25
CLEAR_LOG = "import pathlib; [p.unlink() for p in pathlib.Path('/app/log').glob('predictions.sqlite*')]"

DISCLAIMER = ("This demonstrates the platform DETECTING a shift in traffic and RETRAINING a challenger. It does not show "
              "the model adapting: the challenger is trained on Jan-Mar and validated on April and has never seen the "
              "surge. The +40% durations in the surge window are SYNTHETIC (a made-up ground truth).")


class DemoError(RuntimeError):
    """Something the demo cannot continue without (clear message, no traceback)."""


class LogError(DemoError):
    """The prediction log could not be reset or fetched from the Compose stack."""


# ---- inputs -----------------------------------------------------------------------------------------------------

def load_holdout(path: Path, zones_path: Path) -> tuple[pd.DataFrame, dict]:
    """Clean the held-out month with the project's own cleaning function (in memory; nothing is written)."""
    path = Path(path)
    if not path.exists():
        raise DemoError(f"holdout file not found: {path} (expected a TLC file such as yellow_tripdata_2026-04.parquet)")
    return clean_file(path, Path(zones_path))


def composition(trips: pd.DataFrame) -> dict:
    """Evening-hour share and airport-trip share of a set of trips."""
    return {"evening_share": round(float(_evening(trips).mean()), 4),
            "airport_share": round(float(_airport(trips).mean()), 4)}


def _evening(trips: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(trips["pickup_ts"]).dt.hour.isin(EVENING_HOURS)


def _airport(trips: pd.DataFrame) -> pd.Series:
    return trips["PULocationID"].isin(AIRPORT_ZONES) | trips["DOLocationID"].isin(AIRPORT_ZONES)


def _draw(pool: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    if k <= 0:
        return pool.iloc[0:0]
    if pool.empty:
        raise DemoError("cannot draw the requested trips: the holdout has none of that kind (evening / airport)")
    return pool.sample(k, replace=len(pool) < k, random_state=seed)


def _weekday_counts(shares: pd.Series, k: int) -> pd.Series:
    """k draws split across weekdays in proportion to ``shares`` (largest-remainder rounding, sums to k)."""
    raw = shares * k
    counts = raw.astype(int)
    for day in (raw - counts).sort_values(ascending=False).index[: k - int(counts.sum())]:
        counts[day] += 1
    return counts


def _draw_by_weekday(pool: pd.DataFrame, k: int, seed: int, shares: pd.Series) -> pd.DataFrame:
    """Draw k trips so that their weekday mix follows ``shares`` (the month's own mix) as far as the pool allows."""
    if k <= 0:
        return pool.iloc[0:0]
    if pool.empty:
        raise DemoError("cannot draw the requested trips: the holdout has none of that kind (evening / airport)")
    weekday = pd.to_datetime(pool["pickup_ts"]).dt.dayofweek
    parts = []
    for day, count in _weekday_counts(shares, k).items():
        day_pool = pool[weekday == day]
        if count > 0:
            parts.append(_draw(day_pool if not day_pool.empty else pool, int(count), seed + 7 * int(day)))
    return pd.concat(parts)


def sample_baseline(trips: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """N trips as they are, in pickup order, with the month's own weekday mix.

    The trips are untouched; only the draw is stratified by weekday, so a small window cannot cross the day-of-week
    effect floor by sampling noise alone (the month's own weekday mix is already close to it).
    """
    shares = pd.to_datetime(trips["pickup_ts"]).dt.dayofweek.value_counts(normalize=True).reindex(range(7), fill_value=0.0)
    return _draw_by_weekday(trips, n, seed, shares).sort_values("pickup_ts").reset_index(drop=True)


def sample_surge(trips: pd.DataFrame, n: int, seed: int = 0, evening_share: float = EVENING_SHARE,
                 airport_share: float = AIRPORT_SHARE) -> pd.DataFrame:
    """N trips resampled toward the evening hours with a higher airport share (four cells, drawn separately).

    Within each cell the weekday mix of the month is kept, so only the hour of day and the airport share move: evening
    trips would otherwise skew toward certain weekdays and drag the day-of-week column along.
    """
    evening, airport = _evening(trips), _airport(trips)
    shares = pd.to_datetime(trips["pickup_ts"]).dt.dayofweek.value_counts(normalize=True).reindex(range(7), fill_value=0.0)
    n_evening = round(n * evening_share)
    n_off = n - n_evening
    cells = [(n_evening - round(n_evening * airport_share), evening & ~airport),
             (round(n_evening * airport_share), evening & airport),
             (n_off - round(n_off * airport_share), ~evening & ~airport),
             (round(n_off * airport_share), ~evening & airport)]
    parts = [_draw_by_weekday(trips[mask], k, seed + 100 * i, shares) for i, (k, mask) in enumerate(cells)]
    return pd.concat(parts).sort_values("pickup_ts").reset_index(drop=True)


def build_history(march: pd.DataFrame, april: pd.DataFrame) -> pd.DataFrame:
    """History for the point-in-time rolling features: the last 27h of March plus all of April.

    Without the March tail the first April day would have empty rolling features (a cold start the monthly
    evaluation does not have).
    """
    start = pd.to_datetime(april["pickup_ts"]).min().floor("h")
    tail = march[pd.to_datetime(march["pickup_ts"]) > start - pd.Timedelta(hours=HISTORY_HOURS)]
    return pd.concat([tail, april], ignore_index=True)


# ---- replay through the API -------------------------------------------------------------------------------------

@dataclass
class ReplayStats:
    sent: int = 0
    ok: int = 0
    failed: int = 0
    errors: list = field(default_factory=list)  # the first few messages
    aborted: bool = False  # stopped early after too many failures in a row


def replay(trips: pd.DataFrame, send, max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES) -> ReplayStats:
    """Send every trip to /predict. send(payload) returns the HTTP status; failures are counted, never raised."""
    stats, streak = ReplayStats(), 0
    for row in trips.itertuples():
        payload = {"pickup_zone": int(row.PULocationID), "dropoff_zone": int(row.DOLocationID),
                   "pickup_ts": pd.Timestamp(row.pickup_ts).isoformat()}
        stats.sent += 1
        try:
            status = send(payload)
            error = None if status == 200 else f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001  any transport problem is a failed request, not a crash
            error = f"{type(exc).__name__}: {exc}"
        if error is None:
            stats.ok, streak = stats.ok + 1, 0
            continue
        stats.failed, streak = stats.failed + 1, streak + 1
        if len(stats.errors) < 3:
            stats.errors.append(error)
        if streak >= max_consecutive_failures:
            stats.aborted = True
            break
    return stats


def httpx_sender(client):
    """send(payload) for an httpx.Client whose base_url is the API."""
    return lambda payload: client.post("/predict", json=payload).status_code


class ComposeLog:
    """Reset and fetch the API's prediction log (a SQLite file in the compose `prediction-log` volume).

    Reading a WAL-mode SQLite file from a running container is not safe, so the API is stopped (which flushes and
    closes the log), the file is copied out, and the API is started again.
    """

    def __init__(self, base_url: str, work: Path, run=subprocess.run, get=None, sleep=time.sleep, cwd: Path = ROOT,
                 service: str = "api", log_path: str = "/app/log/predictions.sqlite", timeout: float = 120.0):
        self.base_url, self.work, self.run, self.sleep, self.cwd = base_url.rstrip("/"), Path(work), run, sleep, cwd
        self.service, self.log_path, self.timeout = service, log_path, timeout
        self._get = get

    def _compose(self, *args: str) -> None:
        try:
            self.run(["docker", "compose", *args], check=True, capture_output=True, text=True, cwd=str(self.cwd))
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", "") or exc
            raise LogError(f"`docker compose {' '.join(args)}` failed: {str(detail).strip()[:300]}") from exc

    def _wait_healthy(self) -> None:
        import httpx

        get = self._get or httpx.get
        deadline = time.monotonic() + self.timeout
        while True:
            with contextlib.suppress(Exception):  # not up yet: keep polling until the deadline
                if get(f"{self.base_url}/health", timeout=3).status_code == 200:
                    return
            if time.monotonic() > deadline:
                raise LogError(f"the API did not become healthy at {self.base_url} within {self.timeout:.0f}s")
            self.sleep(2)

    def reset(self) -> None:
        """Empty the log, so each window is measured on its own requests."""
        self._compose("stop", self.service)
        try:
            self._compose("run", "--rm", "--no-deps", "--entrypoint", "python", self.service, "-c", CLEAR_LOG)
        finally:
            self._compose("start", self.service)
        self._wait_healthy()

    def fetch(self, name: str) -> pd.DataFrame:
        """The window's log as a DataFrame."""
        dest = self.work / f"{name}_prediction_log.sqlite"
        for suffix in ("", "-wal", "-shm"):
            Path(str(dest) + suffix).unlink(missing_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self._compose("stop", self.service)
        try:
            self._compose("cp", f"{self.service}:{self.log_path}", str(dest))
        finally:
            self._compose("start", self.service)
        self._wait_healthy()
        return read_log(dest)


# ---- offline scoring ---------------------------------------------------------------------------------------------

def score(model, window: pd.DataFrame, weather: pd.DataFrame, history: pd.DataFrame, factor: float = 1.0) -> dict:
    """Offline MAE of ``model`` on ``window`` with point-in-time rolling features from ``history``.

    factor scales the TRUE durations that the predictions are scored against (1.0 = as they are; 1.4 = the SYNTHETIC
    surge ground truth). The rolling features always come from the true history.
    """
    scored = window.assign(duration_min=window["duration_min"] * factor)
    metrics = evaluate(model, scored, weather, history)
    return {k: float(metrics[k]) for k in ("mae", "rmse", "p90_abs_error")}


def mae_flag(baseline_mae: float, window_mae: float, threshold: float = MAE_FLAG_THRESHOLD) -> bool:
    """True when the window's MAE is MORE than ``threshold`` (relative) above the baseline window's."""
    return window_mae > baseline_mae * (1 + threshold)


def load_champion_readonly(tracking_uri: str | None = None):
    """(model, validation MAE logged for it). Only reads the registry: no alias, version or run is created or moved."""
    from mlflow import MlflowClient

    from training import MODEL_NAME, default_tracking_uri, load_model

    uri = tracking_uri or default_tracking_uri()
    model = load_model("champion", uri)
    version = MlflowClient(uri).get_model_version_by_alias(MODEL_NAME, "champion")
    return model, MlflowClient(uri).get_run(version.run_id).data.metrics.get("mae")


# ---- the retrain, in a throwaway store ---------------------------------------------------------------------------

def assert_throwaway_store(uri: str, work: Path) -> Path:
    """The retrain may only write to a SQLite file inside ``work``: never an http server, never the default store."""
    from training import default_tracking_uri

    if not uri.startswith("sqlite:///"):
        raise DemoError(f"refusing a non-SQLite tracking store for the demo retrain: {uri}")
    path = Path(uri.removeprefix("sqlite:///")).resolve()
    if Path(work).resolve() not in path.parents:
        raise DemoError(f"refusing to write the demo retrain outside {work}: {path}")
    if uri in {default_tracking_uri(), os.environ.get("MLFLOW_TRACKING_URI")}:
        raise DemoError("refusing to use the real tracking store for the demo retrain")
    return path


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def retrain_challenger(drift_report: Path | None, work: Path, rows: int | None, holdout: Path, raw_dir: Path,
                       flow=None) -> dict:
    """Run flows.training_flow.retrain_flow against a throwaway store. Returns what the flow reported.

    Months Jan-Mar are the training months and April (the held-out month) is validation, exactly the monthly retrain
    rule. The throwaway store has no champion, so the flow's new-month rule also fires; that is noted in the result.
    """
    work = Path(work)
    scratch = work / "retrain"
    if scratch.exists():
        shutil.rmtree(scratch)  # only ever inside the demo work dir (asserted below)
    scratch.mkdir(parents=True)
    uri = f"sqlite:///{(scratch / 'mlflow.db').as_posix()}"
    assert_throwaway_store(uri, work)

    raw = scratch / "raw"
    raw.mkdir()
    for name in ("yellow_tripdata_2026-01.parquet", "yellow_tripdata_2026-02.parquet", "yellow_tripdata_2026-03.parquet",
                 "taxi_zone_lookup.csv", "weather_2026_01_07.json"):
        _link_or_copy(Path(raw_dir) / name, raw / name)
    _link_or_copy(Path(holdout), raw / Path(holdout).name)

    from mlflow import MlflowClient

    MlflowClient(uri).create_experiment("eta-duration", artifact_location=(scratch / "artifacts").as_uri())
    if flow is None:
        from flows.training_flow import retrain_flow as flow
    result = flow(drift_report=Path(drift_report) if drift_report else None, raw_dir=raw, processed_dir=scratch / "processed", tracking_uri=uri,
                  max_train_rows=rows, feast_root=scratch / "feast")
    return {"triggered": bool(result.get("retrained")), "reason": result.get("reason"),
            "challenger_version": result.get("version"), "challenger_mae": result.get("mae"),
            "challenger_baseline_mae": result.get("baseline_mae"), "tracking_store": uri, "train_rows": rows or "all",
            "note": ("throwaway store: it has no champion, so the flow's new-month rule fires as well as drift; "
                     "the alias 'champion' that train() bootstraps exists only inside it")}


# ---- the demo ----------------------------------------------------------------------------------------------------

@dataclass
class DemoConfig:
    n: int = DEFAULT_N
    seed: int = 0
    reference_rows: int = DEFAULT_REFERENCE_ROWS  # rows of April scored for the full-month reference MAE (0 = all)
    retrain: bool = True
    retrain_rows: int | None = DEFAULT_RETRAIN_ROWS  # None = a full run
    mae_threshold: float = MAE_FLAG_THRESHOLD
    report: Path = REPORT
    work: Path = WORK


def _drift_part(name: str, log, reference: pd.DataFrame, stats: ReplayStats, work: Path) -> dict:
    """Fetch the window's log and run the drift check. Any failure becomes a status, never an exception."""
    base = {"dataset_drift": None, "trigger_columns": [], "context_columns": [], "drift_status": "not run",
            "drift_report": None}
    if stats.ok == 0:
        return {**base, "drift_status": f"no request succeeded ({'; '.join(stats.errors) or 'none sent'})"}
    try:
        window_log = log.fetch(name)
    except DemoError as exc:
        return {**base, "drift_status": f"log unavailable: {exc}"}
    if window_log.empty:
        return {**base, "drift_status": "the prediction log is empty (is ETA_PREDICTION_LOG set on the api?)"}
    out = Path(work) / name
    out.mkdir(parents=True, exist_ok=True)
    result = check_log_drift(reference, window_log, html_path=out / "drift_report.html")
    (out / "drift_report.json").write_text(json.dumps(result, indent=2, default=str))
    return {"dataset_drift": bool(result["dataset_drift"]), "trigger_columns": list(result["drifted_columns"]),
            "context_columns": list(result["context_drift"]["drifted_columns"]), "drift_status": result["status"],
            "drift_report": str(out / "drift_report.json"), "logged_rows": len(window_log),
            "trigger_tests": result.get("trigger_tests", {})}


def run_window(name: str, label: str, trips: pd.DataFrame, factor: float, *, send, log, reference, model, weather,
               history, work: Path) -> dict:
    """One window: replay through the API, drift from the log, MAE offline."""
    log_ok = True
    try:
        log.reset()
    except DemoError as exc:
        log_ok = False
        stats = ReplayStats(errors=[str(exc)])
    if log_ok:
        stats = replay(trips, send)
    drift = _drift_part(name, log, reference, stats, work) if log_ok else {
        "dataset_drift": None, "trigger_columns": [], "context_columns": [], "drift_report": None,
        "drift_status": f"log unavailable: {stats.errors[0]}"}
    metrics = score(model, trips, weather, history, factor)
    return {"window": name, "label": label, "rows": len(trips), "replayed_ok": stats.ok, "api_failed": stats.failed,
            "replay_aborted": stats.aborted, "api_errors": stats.errors, "composition": composition(trips),
            "mae_source": (f"offline, SYNTHETIC true durations x{factor:.1f}" if factor != 1.0
                           else "offline, true durations"),
            "synthetic": factor != 1.0, **metrics, **drift}


def run_demo(cfg: DemoConfig, *, april: pd.DataFrame, march: pd.DataFrame, reference: pd.DataFrame,
             weather: pd.DataFrame, send, log, champion, retrain) -> dict:
    """Run both windows and the retrain decision. champion() -> (model, march validation MAE); retrain(path) -> dict."""
    model, march_mae = champion()
    history = build_history(march, april)
    sample = april if not cfg.reference_rows or len(april) <= cfg.reference_rows else april.sample(
        cfg.reference_rows, random_state=cfg.seed)
    reference_april = score(model, sample, weather, history)

    common = {"send": send, "log": log, "reference": reference, "model": model, "weather": weather,
              "history": history, "work": cfg.work}
    baseline = run_window("baseline", "baseline (April as-is)", sample_baseline(april, cfg.n, cfg.seed), 1.0, **common)
    surge = run_window("surge", "surge (evening + airports)", sample_surge(april, cfg.n, cfg.seed), SYNTHETIC_FACTOR,
                       **common)

    flags = {"mae_over_threshold": mae_flag(baseline["mae"], surge["mae"], cfg.mae_threshold),
             "mae_change_vs_baseline": surge["mae"] / baseline["mae"] - 1,
             "dataset_drift": bool(surge["dataset_drift"])}
    fired = flags["mae_over_threshold"] or flags["dataset_drift"]
    retrain_result = {"triggered": False, "reason": "no trigger fired" if not fired else "disabled (--no-retrain)",
                      "challenger_version": None}
    if fired and cfg.retrain:
        try:
            drift_path = surge.get("drift_report") or baseline.get("drift_report")
            retrain_result = retrain(Path(drift_path) if drift_path else None)
        except Exception as exc:  # noqa: BLE001  a failed retrain is reported, the summary still prints
            retrain_result = {"triggered": False, "reason": f"retrain failed: {type(exc).__name__}: {exc}",
                              "challenger_version": None}
    surge["retrain_triggered"] = bool(retrain_result.get("triggered"))
    surge["challenger_version"] = retrain_result.get("challenger_version")
    baseline["retrain_triggered"], baseline["challenger_version"] = False, None

    report = {"disclaimer": DISCLAIMER, "synthetic": f"surge-window durations are SYNTHETIC (x{SYNTHETIC_FACTOR:.1f})",
              "settings": {"n": cfg.n, "seed": cfg.seed, "evening_hours": [EVENING_HOURS[0], EVENING_HOURS[-1]],
                           "evening_share": EVENING_SHARE, "airport_share": AIRPORT_SHARE,
                           "mae_flag_threshold": cfg.mae_threshold, "min_rows_for_drift": MIN_ROWS},
              "mae_reference": {"champion_validation_mae_march": march_mae,
                                "april_full_month_mae": reference_april["mae"],
                                "april_reference_rows": len(sample),
                                "note": "the monthly figure comes from the registry run; the April one is scored here "
                                        "with the same evaluation code"},
              "windows": [baseline, surge], "flags": flags, "retrain": retrain_result}
    Path(cfg.report).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.report).write_text(json.dumps(report, indent=2, default=str))
    return report


def format_summary(report: dict) -> str:
    """The summary table plus the plain-language caveats."""
    headers = ["window", "rows", "MAE source", "MAE", "dataset_drift", "trigger cols drifted", "context cols",
               "retrain", "challenger"]
    rows = []
    for w in report["windows"]:
        drift = "n/a" if w["dataset_drift"] is None else str(w["dataset_drift"]).lower()
        note = "" if w["dataset_drift"] is not None else f" ({w['drift_status']})"
        rows.append([w["label"], str(w["rows"]) + (f" ({w['api_failed']} failed)" if w["api_failed"] else ""),
                     w["mae_source"], f"{w['mae']:.3f}", drift + note, ", ".join(w["trigger_columns"]) or "-",
                     ", ".join(w["context_columns"]) or "-", "yes" if w["retrain_triggered"] else "no",
                     w["challenger_version"] or "-"])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True))

    ref, flags = report["mae_reference"], report["flags"]
    march = ref["champion_validation_mae_march"]
    march_text = f"{march:.3f}" if march is not None else "n/a"
    threshold = report["settings"]["mae_flag_threshold"] * 100
    out = [line(headers), "-+-".join("-" * w for w in widths), *(line(r) for r in rows), "",
           (f"MAE reference: champion validation MAE on March = {march_text} | full-month April MAE here = "
            f"{ref['april_full_month_mae']:.3f} ({ref['april_reference_rows']} rows)"),
           (f"MAE flag (> +{threshold:.0f}% over baseline): {'FIRED' if flags['mae_over_threshold'] else 'no'} "
            f"(surge MAE is {flags['mae_change_vs_baseline'] * 100:+.1f}% vs baseline) | "
            f"drift flag: {'FIRED' if flags['dataset_drift'] else 'no'}"),
           f"Retrain: {report['retrain'].get('reason')}", "", report["disclaimer"]]
    return chr(10).join(out)


def main(argv=None, *, send=None, log=None, champion=None, retrain=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000", help="the API of the compose stack")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="requests per window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=Path, default=HOLDOUT)
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--tracking-uri", default=None,
                        help="registry to READ the champion from (default: MLFLOW_TRACKING_URI)")
    parser.add_argument("--reference-rows", type=int, default=DEFAULT_REFERENCE_ROWS,
                        help="April rows for the full-month MAE (0 = all)")
    parser.add_argument("--retrain-rows", type=int, default=DEFAULT_RETRAIN_ROWS,
                        help="training-row sample for the quick retrain")
    parser.add_argument("--full-retrain", action="store_true", help="train on every row (slow)")
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--mae-threshold", type=float, default=MAE_FLAG_THRESHOLD)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args(argv)

    client = None
    try:
        april, _ = load_holdout(args.holdout, args.raw_dir / "taxi_zone_lookup.csv")
        trips = pd.read_parquet(args.trips)
        weather = load_weather(args.raw_dir / "weather_2026_01_07.json")
        cfg = DemoConfig(n=args.n, seed=args.seed, reference_rows=args.reference_rows, retrain=not args.no_retrain,
                         retrain_rows=None if args.full_retrain else args.retrain_rows, mae_threshold=args.mae_threshold,
                         report=args.report, work=args.work)
        if send is None:
            import httpx

            client = httpx.Client(base_url=args.url, timeout=10)
            send = httpx_sender(client)
        log = log or ComposeLog(args.url, args.work)
        champion = champion or (lambda: load_champion_readonly(args.tracking_uri))

        def default_retrain(drift_path):
            return retrain_challenger(drift_path, args.work, cfg.retrain_rows, args.holdout, args.raw_dir)

        report = run_demo(cfg, april=april, march=trips[trips["month"] == "2026-03"],
                          reference=reference_features(trips, weather), weather=weather, send=send, log=log,
                          champion=champion, retrain=retrain or default_retrain)
    except DemoError as exc:
        print(f"surge demo cannot run: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            client.close()
    print(format_summary(report))
    print(f"\nreport written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
