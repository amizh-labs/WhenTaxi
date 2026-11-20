"""Two-fold comparison of model variants (feature and objective experiments), no MLflow side effects.

Folds (time-ordered, never random): A = train Jan / validate Feb, B = train Jan-Feb / validate Mar.
Variants: current (12 features) | +geo features | +geo +L1 | +geo +Huber | +geo +log1p target.
Run: python -m evaluation.model_pass [--max-train-rows 2000000] [--out data/processed/model_pass.json]
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import (
    FEATURE_COLUMNS,
    GEO_COLUMNS,
    ROLLING_COLUMNS,
    build_features,
    load_centroids,
    load_weather,
    load_zone_categories,
    rolling_pair_features,
)
from training.metrics import all_metrics
from training.train import LGB_PARAMS

ROOT = Path(__file__).resolve().parents[2]
TARGET = "duration_min"
FOLDS = {"A": (["2026-01"], ["2026-02"]), "B": (["2026-01", "2026-02"], ["2026-03"])}
BASE_COLS = FEATURE_COLUMNS + ROLLING_COLUMNS
HUBER_DELTA = 5.0  # minutes: residuals beyond this are treated linearly
VARIANTS = {  # name -> (columns, lightgbm params, log1p target)
    "current": (BASE_COLS, {}, False),
    "+geo features": (BASE_COLS + GEO_COLUMNS, {}, False),
    "+geo +L1": (BASE_COLS + GEO_COLUMNS, {"objective": "l1"}, False),
    "+geo +Huber": (BASE_COLS + GEO_COLUMNS, {"objective": "huber", "alpha": HUBER_DELTA}, False),
    "+geo +log1p": (BASE_COLS + GEO_COLUMNS, {}, True),
}


def fold_data(trips, weather, zones, train_months, valid_months, max_rows, seed=0):
    tr, va = trips[trips["month"].isin(train_months)], trips[trips["month"].isin(valid_months)]
    if max_rows and len(tr) > max_rows:
        tr = tr.sample(max_rows, random_state=seed)
    x_tr = build_features(tr, weather, zones, rolling_pair_features(trips, tr), geo=True)
    x_va = build_features(va, weather, zones, rolling_pair_features(trips, va), geo=True)
    return x_tr, x_va


def fit_predict(x_tr, x_va, columns, params, log_target, seed=0):
    model = lgb.LGBMRegressor(**{**LGB_PARAMS, **params, "random_state": seed})
    model.fit(x_tr[columns], np.log1p(x_tr[TARGET]) if log_target else x_tr[TARGET])
    raw = model.predict(x_va[columns])
    return np.expm1(raw) if log_target else raw


def breakdowns(x_va: pd.DataFrame, pred: np.ndarray) -> dict:
    """MAE by pickup hour, pickup borough and airport / non-airport trips."""
    err = pd.Series(np.abs(x_va[TARGET].to_numpy() - pred), index=x_va.index)
    borough = (x_va["borough_pair"] // 10).map(dict(enumerate(["Bronx", "Brooklyn", "EWR", "Manhattan", "Queens",
                                                               "Staten Island", "Unknown"])))
    airport = x_va["airport_flag"].map({1: "airport", 0: "non-airport"})
    out = {}
    for name, key in (("hour", x_va["hour"]), ("borough", borough), ("airport", airport)):
        g = err.groupby(key.to_numpy())
        out[name] = {str(k): {"mae": float(v.mean()), "trips": len(v)} for k, v in g}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    p.add_argument("--weather", type=Path, default=ROOT / "data" / "raw" / "weather_2026_01_07.json")
    p.add_argument("--zones", type=Path, default=ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    p.add_argument("--max-train-rows", type=int, default=2_000_000)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "model_pass.json")
    args = p.parse_args(argv)
    load_centroids()
    trips, weather, zones = pd.read_parquet(args.trips), load_weather(args.weather), load_zone_categories(args.zones)
    start = time.time()
    results: dict = {"folds": {}, "max_train_rows": args.max_train_rows}
    preds: dict = {}
    for fold, (tm, vm) in FOLDS.items():
        x_tr, x_va = fold_data(trips, weather, zones, tm, vm, args.max_train_rows)
        results["folds"][fold] = {}
        for name, (cols, params, log_target) in VARIANTS.items():
            t0 = time.time()
            pred = fit_predict(x_tr, x_va, cols, params, log_target)
            m = all_metrics(x_va[TARGET], pred)
            results["folds"][fold][name] = {**{k: float(v) for k, v in m.items()}, "seconds": round(time.time() - t0, 1)}
            print(fold, name, {k: round(float(v), 4) for k, v in m.items()}, flush=True)
            if fold == "B":
                preds[name] = pred
        if fold == "B":
            xb = x_va
    results["mean"] = {n: {k: float(np.mean([results["folds"][f][n][k] for f in FOLDS]))
                           for k in ("mae", "rmse", "p90_abs_error")} for n in VARIANTS}
    best = min(results["mean"], key=lambda n: results["mean"][n]["mae"])
    results["best"] = best
    results["breakdown_fold_B"] = {"current": breakdowns(xb, preds["current"]), best: breakdowns(xb, preds[best])}
    results["total_seconds"] = round(time.time() - start, 1)
    args.out.write_text(json.dumps(results, indent=2))
    print("best:", best, "total seconds:", results["total_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
