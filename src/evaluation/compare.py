"""Evaluate registered models on a held-out month and compare champion vs challenger."""
import pandas as pd

from features import build_features, rolling_pair_features
from training.metrics import all_metrics

TARGET = "duration_min"


def predict_frame(model, trips: pd.DataFrame, weather: pd.DataFrame, history: pd.DataFrame | None = None
                  ) -> pd.DataFrame:
    """Trips plus weather columns, prediction and absolute error (feature code is the shared one).

    history: completed trips used for rolling zone-pair features (defaults to ``trips`` itself;
    pass earlier months too so the start of the evaluated month is not cold).
    """
    rolling = rolling_pair_features(trips if history is None else history, trips) if model.needs_rolling else None
    features = build_features(trips, weather, model.zone_categories, rolling)
    out = trips.copy()
    for col in ("temperature_2m", "precipitation", "snowfall", "wind_speed_10m"):
        out[col] = features[col]
    out["hour"] = features["hour"]
    out["prediction"] = model.predict(features)
    out["abs_error"] = (out[TARGET] - out["prediction"]).abs()
    return out


def evaluate(model, trips: pd.DataFrame, weather: pd.DataFrame, history: pd.DataFrame | None = None
             ) -> dict[str, float]:
    scored = predict_frame(model, trips, weather, history)
    return all_metrics(scored[TARGET], scored["prediction"])


def compare(champion, challenger, trips: pd.DataFrame, weather: pd.DataFrame,
            history: pd.DataFrame | None = None) -> dict:
    champ = evaluate(champion, trips, weather, history)
    chall = evaluate(challenger, trips, weather, history)
    return {
        "champion": {"version": champion.version, **champ},
        "challenger": {"version": challenger.version, **chall},
        "mae_relative_change": (chall["mae"] - champ["mae"]) / champ["mae"],
        "rows": len(trips),
    }
