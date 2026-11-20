"""Train the baseline and LightGBM, track in MLflow, register the model.

Time split: train on the train months, validate on the later month. Never random.
Features and the fixed zone category list come from the shared features module, so
training, evaluation and serving encode zones identically.
"""
import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from features import (
    FEATURE_COLUMNS,
    GEO_COLUMNS,
    ROLLING_COLUMNS,
    WEATHER_COLUMNS,
    build_features,
    load_weather,
    load_zone_categories,
    rolling_pair_features,
)
from training.baseline import ZonePairMeanBaseline
from training.metrics import all_metrics

MODEL_NAME = "eta-duration"
TARGET = "duration_min"
TRAIN_MONTHS = ["2026-01", "2026-02"]
VALID_MONTHS = ["2026-03"]
# Tuned by time-ordered GridSearchCV on Jan-Feb (see experiment notes): small but consistent gain.
LGB_PARAMS = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 127, "min_child_samples": 50,
              "n_jobs": -1, "verbose": -1}
ROOT = Path(__file__).resolve().parents[2]


def default_tracking_uri() -> str:
    """MLFLOW_TRACKING_URI if set, otherwise a local sqlite file in the repo root."""
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def time_split(trips: pd.DataFrame, train_months, valid_months):
    return trips[trips["month"].isin(train_months)], trips[trips["month"].isin(valid_months)]


def train(trips, weather, zone_categories, tracking_uri=None, clean_report=None, train_months=None,
          valid_months=None, max_train_rows=None, params=None, seed=0, use_rolling=True,
          use_geo=False, log_target=False):
    """Returns dict with run_id, model version, metrics and validation weather NaN rates.

    use_geo: add features.geo columns. log_target: fit log1p(duration), predictions are expm1'd (the loaded model does it).
    """
    tracking_uri = tracking_uri or default_tracking_uri()
    train_months = train_months or TRAIN_MONTHS
    valid_months = valid_months or VALID_MONTHS
    train_df, valid_df = time_split(trips, train_months, valid_months)
    if max_train_rows and len(train_df) > max_train_rows:
        train_df = train_df.sample(max_train_rows, random_state=seed)
    if train_df.empty or valid_df.empty:
        raise ValueError("train or validation split is empty")

    baseline = ZonePairMeanBaseline().fit(train_df)
    base_metrics = all_metrics(valid_df[TARGET], baseline.predict(valid_df), prefix="baseline_")

    # Rolling zone-pair stats use ALL given trips as history; point-in-time rules make it leak-free.
    columns = FEATURE_COLUMNS + (ROLLING_COLUMNS if use_rolling else []) + (GEO_COLUMNS if use_geo else [])
    x_train = build_features(train_df, weather, zone_categories,
                             rolling_pair_features(trips, train_df) if use_rolling else None, geo=use_geo)
    x_valid = build_features(valid_df, weather, zone_categories,
                             rolling_pair_features(trips, valid_df) if use_rolling else None, geo=use_geo)
    nan_rates = {f"valid_nan_rate_{c}": float(x_valid[c].isna().mean()) for c in WEATHER_COLUMNS}

    params = {**LGB_PARAMS, **(params or {}), "random_state": seed}
    model = lgb.LGBMRegressor(**params)
    model.fit(x_train[columns], np.log1p(x_train[TARGET]) if log_target else x_train[TARGET])
    raw = model.predict(x_valid[columns])
    metrics = all_metrics(valid_df[TARGET], np.expm1(raw) if log_target else raw)
    metrics.update(base_metrics)
    metrics["mae_improvement_vs_baseline"] = metrics["baseline_mae"] - metrics["mae"]
    importance = dict(zip(columns, (float(v) for v in model.booster_.feature_importance("gain")), strict=True))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("eta-duration")
    with mlflow.start_run() as run:
        mlflow.log_params({**params, "train_rows": len(train_df), "valid_rows": len(valid_df),
                           "use_rolling": use_rolling, "use_geo": use_geo, "log_target": log_target})
        mlflow.set_tags({"train_months": ",".join(train_months), "valid_months": ",".join(valid_months)})
        mlflow.log_metrics({**metrics, **nan_rates})
        mlflow.log_dict(importance, "feature_importance.json")
        mlflow.log_dict(list(zone_categories), "zone_categories.json")
        if clean_report is not None:
            mlflow.log_dict(clean_report, "clean_report.json")
        info = mlflow.lightgbm.log_model(model, name="model", registered_model_name=MODEL_NAME)
    version = str(info.registered_model_version)

    client = MlflowClient(tracking_uri)
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
    except mlflow.exceptions.MlflowException:
        # Bootstrap: with no champion yet the first model serves. Later promotion is evaluation's job.
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
    return {"run_id": run.info.run_id, "version": version, "metrics": metrics, "nan_rates": nan_rates}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--weather", type=Path, default=ROOT / "data" / "raw" / "weather_2026_01_07.json")
    parser.add_argument("--zones", type=Path, default=ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    parser.add_argument("--clean-report", type=Path, default=ROOT / "data" / "processed" / "data_quality_report.json")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--geo", action="store_true", help="add the static zone-pair geography features")
    parser.add_argument("--objective", default=None, help="LightGBM objective, e.g. l1 (default: regression/l2)")
    parser.add_argument("--log-target", action="store_true", help="fit log1p(duration)")
    args = parser.parse_args()
    report = json.loads(args.clean_report.read_text()) if args.clean_report.exists() else None
    result = train(pd.read_parquet(args.trips), load_weather(args.weather), load_zone_categories(args.zones),
                   clean_report=report, max_train_rows=args.max_train_rows,
                   params={"objective": args.objective} if args.objective else None,
                   use_geo=args.geo, log_target=args.log_target)
    print("weather NaN rate on validation set:")
    for name, rate in result["nan_rates"].items():
        print(f"  {name.removeprefix('valid_nan_rate_')}: {rate:.4%}")
    print({k: result[k] for k in ("run_id", "version", "metrics")})
