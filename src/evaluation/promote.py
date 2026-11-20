"""Promotion logic: move the `champion` alias to the challenger only if it passes thresholds.

Rules (all must hold):
  * challenger MAE is at least MIN_MAE_IMPROVEMENT (relative) better than the champion's
  * challenger p90 absolute error is no more than P90_TOLERANCE (relative) worse
Both are measured on the same held-out month. The decision is written to MLflow as model-version
tags and as an evaluation_report.json artifact on the challenger's run.
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from evaluation.compare import compare, predict_frame
from evaluation.errors import error_analysis, to_jsonable
from features import load_weather
from training import MODEL_NAME, default_tracking_uri, load_model

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Thresholds:
    min_mae_improvement: float = 0.005  # challenger must be >= 0.5% better on MAE
    p90_tolerance: float = 0.02  # ... and p90 may not regress by more than 2%


def decide(comparison: dict, thresholds: Thresholds = Thresholds()) -> tuple[bool, list[str]]:
    champ, chall = comparison["champion"], comparison["challenger"]
    reasons = []
    mae_gain = (champ["mae"] - chall["mae"]) / champ["mae"]
    if mae_gain < thresholds.min_mae_improvement:
        reasons.append(f"MAE improvement {mae_gain:.2%} < required {thresholds.min_mae_improvement:.2%}")
    p90_change = (chall["p90_abs_error"] - champ["p90_abs_error"]) / champ["p90_abs_error"]
    if p90_change > thresholds.p90_tolerance:
        reasons.append(f"p90 error regressed {p90_change:.2%} > allowed {thresholds.p90_tolerance:.2%}")
    return not reasons, reasons


def promote_if_better(trips: pd.DataFrame, weather: pd.DataFrame, tracking_uri: str | None = None,
                      thresholds: Thresholds = Thresholds(), dry_run: bool = False,
                      history: pd.DataFrame | None = None) -> dict:
    uri = tracking_uri or default_tracking_uri()
    champion, challenger = load_model("champion", uri), load_model("challenger", uri)
    if champion.version == challenger.version:
        return {"promoted": False, "reasons": ["challenger is already the champion"], "version": champion.version}

    comparison = compare(champion, challenger, trips, weather, history)
    ok, reasons = decide(comparison, thresholds)
    report = {"promoted": ok and not dry_run, "reasons": reasons or ["passed all thresholds"],
              "thresholds": asdict(thresholds), "comparison": comparison,
              "challenger_error_analysis": to_jsonable(error_analysis(predict_frame(challenger, trips, weather, history)))}

    client = MlflowClient(uri)
    version = client.get_model_version(MODEL_NAME, challenger.version)
    client.log_dict(version.run_id, report, "evaluation_report.json")
    client.set_model_version_tag(MODEL_NAME, challenger.version, "promotion",
                                 "promoted" if report["promoted"] else "rejected")
    if report["promoted"]:
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger.version)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2026-03", help="held-out month to evaluate on")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trips", type=Path, default=ROOT / "data" / "processed" / "trips_clean.parquet")
    parser.add_argument("--weather", type=Path, default=ROOT / "data" / "raw" / "weather_2026_01_07.json")
    args = parser.parse_args()
    trips = pd.read_parquet(args.trips)
    report = promote_if_better(trips[trips["month"] == args.month], load_weather(args.weather),
                               dry_run=args.dry_run, history=trips)
    print(json.dumps({k: report[k] for k in ("promoted", "reasons")}, indent=2))
