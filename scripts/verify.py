"""Sanity checks for the modeling rules: prints PASS/FAIL with numbers for each, exit code 1 if any fails.

    python scripts/verify.py                         # real data: data/processed + data/raw (takes a few minutes)
    python scripts/verify.py --trips t.parquet --weather w.json --zones z.csv --max-train-rows 100000

Checks (all independent of your MLflow registry: training runs against a throw-away SQLite store):
  1. leakage     no trip_distance / fare_amount / total_amount / tip_amount in the feature list or the model
  2. time-split  max train pickup is earlier than min validation pickup
  3. weather     weather NaN rate after the join is under 1% on the validation set
  4. baseline    LightGBM MAE is below the zone-pair-mean baseline MAE
  5. seed        two training runs with the same seed give the same MAE
  6. unseen-zone a prediction for a zone absent from the training months returns a number
"""
import argparse
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

import pandas as pd
import prometheus_client  # noqa: F401  (before feast: importing feast sets PROMETHEUS_MULTIPROC_DIR)

from evaluation.gate import REAL_PARAMS, REAL_TRAIN_ROWS, ZONES
from features import (
    FEATURE_COLUMNS,
    GEO_COLUMNS,
    ROLLING_COLUMNS,
    WEATHER_COLUMNS,
    build_features,
    load_weather,
    load_zone_categories,
)
from training import load_model, train
from training.train import TRAIN_MONTHS, VALID_MONTHS

FORBIDDEN = ("trip_distance", "fare_amount", "total_amount", "tip_amount")  # unknown at request time = leakage
MAX_NAN_RATE = 0.01
SEED = 0
UNSEEN_FALLBACK_ZONE = 9999  # used only when every real zone appears in the training months


class Fail(Exception):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


class Context:
    """Loads the inputs once and trains lazily (checks 1, 3-6 share the same two training runs)."""

    def __init__(self, args):
        self.trips = pd.read_parquet(args.trips)
        self.train_months = args.train_months
        self.valid_months = args.valid_months
        self.max_train_rows = args.max_train_rows
        self.zones = load_zone_categories(args.zones) if args.zones and args.zones.exists() else list(ZONES)
        if args.weather and args.weather.exists():
            self.weather, self.weather_note = load_weather(args.weather), ""
        else:
            hours = pd.date_range(self.trips["pickup_ts"].min().floor("D"),
                                  self.trips["pickup_ts"].max().ceil("D") + pd.Timedelta(days=1), freq="h")
            self.weather = pd.DataFrame({"weather_hour": hours, **{c: 0.0 for c in WEATHER_COLUMNS}})
            self.weather_note = " (synthetic zero weather: no weather file given)"
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # sqlite stays locked on Windows
        self.uri = f"sqlite:///{(Path(self._tmp.name) / 'mlflow.db').as_posix()}"
        self._runs: list[dict] = []

    def split(self):
        in_train = self.trips["month"].isin(self.train_months)
        in_valid = self.trips["month"].isin(self.valid_months)
        return self.trips[in_train], self.trips[in_valid]

    def run(self, index: int) -> dict:
        while len(self._runs) <= index:
            self._runs.append(train(self.trips, self.weather, self.zones, tracking_uri=self.uri,
                                    train_months=self.train_months, valid_months=self.valid_months,
                                    max_train_rows=self.max_train_rows, params=REAL_PARAMS, seed=SEED,
                                    use_geo=True))  # the shipped configuration: geo features + L1 objective
        return self._runs[index]

    def close(self):
        self._tmp.cleanup()


# ---- checks: return a message with the numbers, or raise Fail ----------------------------------

def check_leakage(ctx: Context) -> str:
    declared = FEATURE_COLUMNS + ROLLING_COLUMNS + GEO_COLUMNS  # every column any model variant may use
    ctx.run(0)  # trains (or reuses) the run whose model is inspected below
    model = load_model("champion", tracking_uri=ctx.uri)
    trained = model.feature_columns
    bad = sorted(set(FORBIDDEN) & (set(declared) | set(trained)))
    need(not bad, f"leaking columns in the feature list/model: {bad}")
    return f"{len(trained)} model features, none of {list(FORBIDDEN)}: {trained}"


def check_time_split(ctx: Context) -> str:
    train_df, valid_df = ctx.split()
    need(len(train_df) > 0 and len(valid_df) > 0, f"empty split: train {len(train_df)}, validation {len(valid_df)} rows")
    last_train, first_valid = train_df["pickup_ts"].max(), valid_df["pickup_ts"].min()
    need(last_train < first_valid, f"max train pickup {last_train} is not before min validation pickup {first_valid}")
    return (f"max train pickup {last_train:%Y-%m-%d %H:%M} < min validation pickup {first_valid:%Y-%m-%d %H:%M} "
            f"(train {len(train_df):,} rows {ctx.train_months}, validation {len(valid_df):,} rows {ctx.valid_months})")


def check_weather(ctx: Context) -> str:
    rates = ctx.run(0)["nan_rates"]
    worst = max(rates.values())
    detail = ", ".join(f"{k.removeprefix('valid_nan_rate_')} {v:.4%}" for k, v in rates.items())
    need(worst < MAX_NAN_RATE, f"weather NaN rate {worst:.4%} >= {MAX_NAN_RATE:.0%} on validation: {detail}")
    return f"max NaN rate {worst:.4%} (limit {MAX_NAN_RATE:.0%}) on validation: {detail}{ctx.weather_note}"


def check_beats_baseline(ctx: Context) -> str:
    m = ctx.run(0)["metrics"]
    need(m["mae"] < m["baseline_mae"], f"LightGBM MAE {m['mae']:.4f} is not below baseline MAE {m['baseline_mae']:.4f}")
    gain = 100 * (m["baseline_mae"] - m["mae"]) / m["baseline_mae"]
    return (f"LightGBM MAE {m['mae']:.4f} < zone-pair baseline MAE {m['baseline_mae']:.4f} ({gain:.1f}% better; "
            f"RMSE {m['rmse']:.3f}, p90 {m['p90_abs_error']:.3f})")


def check_same_seed_same_mae(ctx: Context) -> str:
    first, second = ctx.run(0)["metrics"]["mae"], ctx.run(1)["metrics"]["mae"]
    need(first == second, f"seed {SEED} gave MAE {first!r} then {second!r} (diff {abs(first - second):.3e})")
    return f"seed {SEED} twice -> MAE {first:.6f} and {second:.6f} (identical)"


def check_unseen_zone(ctx: Context) -> str:
    train_df, valid_df = ctx.split()
    seen = set(train_df["PULocationID"]) | set(train_df["DOLocationID"])
    absent = sorted(set(ctx.zones) - seen)
    zone = absent[0] if absent else UNSEEN_FALLBACK_ZONE
    kind = f"zone {zone} (one of {len(absent)} lookup zones absent from {ctx.train_months})" if absent else \
        f"zone {zone} (every lookup zone appears in training, so an id outside the lookup)"
    model = load_model("champion", tracking_uri=ctx.uri)
    row = pd.DataFrame({"pickup_ts": [valid_df["pickup_ts"].min()], "PULocationID": [zone],
                        "DOLocationID": [int(train_df["DOLocationID"].iloc[0])]})
    rolling = pd.DataFrame([{"avg_duration_1h": float("nan"), "avg_duration_24h": float("nan"),
                             "trip_count_24h": 0.0}], index=row.index) if model.needs_rolling else None
    eta = float(model.predict(build_features(row, ctx.weather, model.zone_categories, rolling, geo=model.needs_geo)).iloc[0])
    need(math.isfinite(eta), f"prediction for {kind} is {eta}")
    return f"{kind}: predicted {eta:.2f} min"


CHECKS = {
    "leakage": check_leakage,
    "time-split": check_time_split,
    "weather": check_weather,
    "baseline": check_beats_baseline,
    "seed": check_same_seed_same_mae,
    "unseen-zone": check_unseen_zone,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--weather", type=Path, default=ROOT / "data" / "raw" / "weather_2026_01_07.json")
    parser.add_argument("--zones", type=Path, default=ROOT / "data" / "raw" / "taxi_zone_lookup.csv")
    parser.add_argument("--train-months", nargs="+", default=TRAIN_MONTHS)
    parser.add_argument("--valid-months", nargs="+", default=VALID_MONTHS)
    parser.add_argument("--max-train-rows", type=int, default=REAL_TRAIN_ROWS)
    args = parser.parse_args(argv)

    ctx = Context(args)
    failed = 0
    try:
        for name, check in CHECKS.items():
            try:
                print(f"PASS  {name:<12} {check(ctx)}", flush=True)
            except Fail as exc:
                failed += 1
                print(f"FAIL  {name:<12} {exc}", flush=True)
            except Exception as exc:  # noqa: BLE001  a crashing check is a failed check, never a skipped one
                failed += 1
                print(f"FAIL  {name:<12} {type(exc).__name__}: {exc}", flush=True)
    finally:
        ctx.close()
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
