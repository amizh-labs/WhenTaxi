"""Prefect flows: end-to-end training and drift/new-month triggered retraining.

Run training:  python -m flows.training_flow
Run retrain:   python -m flows.training_flow --retrain [--drift-report path]
"""
import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, get_run_logger, task

from features import load_weather, load_zone_categories
from features.store import build_online_store
from flows.retrain import read_drift, should_retrain
from training import MODEL_NAME, default_tracking_uri, train
from validation import run as run_cleaning

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MONTHS = ["2026-01", "2026-02", "2026-03"]
SHIPPED_PARAMS = {"objective": "l1"}  # with use_geo=True: the geo + L1 configuration the CI gate and demo use


@task(name="clean-and-validate")
def clean_task(raw_dir: Path, out_dir: Path, months: list[str]) -> list[dict]:
    return run_cleaning(raw_dir, out_dir, months)


@task(name="train-and-register")
def train_task(trips_path: Path, raw_dir: Path, reports: list[dict], train_months, valid_months,
               tracking_uri: str, max_train_rows: int | None) -> dict:
    result = train(
        pd.read_parquet(trips_path),
        load_weather(raw_dir / "weather_2026_01_07.json"),
        load_zone_categories(raw_dir / "taxi_zone_lookup.csv"),
        tracking_uri=tracking_uri,
        clean_report=reports,
        train_months=train_months,
        valid_months=valid_months,
        max_train_rows=max_train_rows,
        params=SHIPPED_PARAMS,
        use_geo=True,
    )
    return {"run_id": result["run_id"], "version": result["version"], "mae": result["metrics"]["mae"],
            "baseline_mae": result["metrics"]["baseline_mae"]}


@task(name="materialize-online-features")
def materialize_task(trips_path: Path, feast_root: Path) -> str:
    """Refresh the Feast online store the API reads (latest rolling zone-pair features)."""
    trips = pd.read_parquet(trips_path)
    upto = trips["pickup_ts"].max().ceil("h")
    build_online_store(trips, feast_root, upto)
    return str(upto)


@flow(name="eta-training")
def training_flow(months: list[str] | None = None, raw_dir: Path = ROOT / "data" / "raw",
                  processed_dir: Path = ROOT / "data" / "processed", tracking_uri: str | None = None,
                  max_train_rows: int | None = None, feast_root: Path = ROOT / "data" / "feast") -> dict:
    """ingest -> clean/validate -> features+train -> register -> refresh online features.

    The last month is validation.
    """
    months = sorted(months or DEFAULT_MONTHS)
    reports = clean_task(raw_dir, processed_dir, months)
    result = train_task(processed_dir / "trips_clean.parquet", raw_dir, reports, months[:-1], months[-1:],
                        tracking_uri or default_tracking_uri(), max_train_rows)
    result["online_features_as_of"] = materialize_task(processed_dir / "trips_clean.parquet", feast_root)
    return result


def champion_months(tracking_uri: str) -> list[str]:
    """Months the current champion was trained and validated on (from its MLflow run tags)."""
    client = MlflowClient(tracking_uri)
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        return []
    tags = client.get_run(version.run_id).data.tags
    return sorted(set(tags["train_months"].split(",")) | set(tags["valid_months"].split(",")))


def available_months(raw_dir: Path) -> list[str]:
    return sorted(p.stem.split("_")[-1] for p in raw_dir.glob("yellow_tripdata_*.parquet"))


@flow(name="eta-retrain-check")
def retrain_flow(drift_report: Path | None = None, raw_dir: Path = ROOT / "data" / "raw",
                 processed_dir: Path = ROOT / "data" / "processed", tracking_uri: str | None = None,
                 max_train_rows: int | None = None, feast_root: Path = ROOT / "data" / "feast") -> dict:
    """Retrain when a new month of data exists or drift was flagged; otherwise do nothing."""
    log = get_run_logger()
    tracking_uri = tracking_uri or default_tracking_uri()
    months = available_months(raw_dir)
    decision, reason = should_retrain(months, champion_months(tracking_uri), read_drift(drift_report))
    log.info("retrain=%s (%s)", decision, reason)
    if not decision:
        return {"retrained": False, "reason": reason}
    result = training_flow(months, raw_dir, processed_dir, tracking_uri, max_train_rows, feast_root)
    return {"retrained": True, "reason": reason, **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--drift-report", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    args = parser.parse_args()
    out = (retrain_flow(args.drift_report, max_train_rows=args.max_train_rows) if args.retrain
           else training_flow(max_train_rows=args.max_train_rows))
    print(json.dumps(out, indent=2))
