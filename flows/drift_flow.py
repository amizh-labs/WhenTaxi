"""Daily drift check on the prediction log (Prefect).

Run once:   python -m flows.drift_flow [--log data/prediction_log.sqlite] [--retrain]
Daily 06:00 (keeps running; needs a Prefect server or the local ephemeral one):
            python -m flows.drift_flow --serve

Reads the prediction log (docs README 'Prediction log'), compares it with the Jan-Feb reference
(monitoring.prediction_drift), writes monitoring/reports/drift_report.json and returns the report. With --retrain,
a drifted window (dataset_drift is true only for hour / day-of-week / distance, never for weather) is handed to
the existing retrain flow, which decides together with the new-month rule.
"""
import argparse
import json
import os
from pathlib import Path

from monitoring.prediction_drift import MIN_ROWS
from monitoring.prediction_drift import run as run_drift
from prefect import flow, get_run_logger, task

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = Path(os.environ.get("ETA_PREDICTION_LOG", ROOT / "data" / "prediction_log.sqlite"))
DEFAULT_OUT = ROOT / "monitoring" / "reports"
CRON = "0 6 * * *"  # every day at 06:00


@task(name="prediction-drift-report")
def drift_task(log_path: Path, trips_path: Path, weather_path: Path, out: Path, min_rows: int) -> dict:
    return run_drift(log_path, trips_path, weather_path, out, min_rows)


@flow(name="eta-daily-drift")
def drift_flow(log_path: Path = DEFAULT_LOG, out: Path = DEFAULT_OUT, retrain: bool = False,
               trips_path: Path = ROOT / "data" / "processed" / "trips_clean.parquet",
               weather_path: Path = ROOT / "data" / "raw" / "weather_2026_01_07.json", min_rows: int = MIN_ROWS) -> dict:
    log = get_run_logger()
    report = drift_task(Path(log_path), Path(trips_path), Path(weather_path), Path(out), min_rows)
    log.info("drift status=%s dataset_drift=%s drifted=%s context=%s data_quality=%s", report["status"],
             report["dataset_drift"], report["drifted_columns"], report["context_drift"]["drifted_columns"],
             report["data_quality"])
    if report["data_quality"]["alert"]:
        log.warning("data quality alert: %.0f%% of logged requests have missing weather",
                    100 * report["data_quality"]["share_missing_weather"])
    outcome = {"report": report, "retrain": None}
    if retrain and report["dataset_drift"]:
        from flows.training_flow import retrain_flow  # imported lazily: it pulls in feast and mlflow

        outcome["retrain"] = retrain_flow(Path(out) / "drift_report.json")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--serve", action="store_true", help=f"run on a schedule (cron '{CRON}') instead of once")
    args = parser.parse_args()
    if args.serve:
        drift_flow.serve(name="eta-daily-drift", cron=CRON,
                         parameters={"log_path": str(args.log), "out": str(args.out), "retrain": args.retrain})
    else:
        print(json.dumps(drift_flow(args.log, args.out, args.retrain), indent=2, default=str))
