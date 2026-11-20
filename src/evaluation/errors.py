"""Error analysis: where does the model do worst? By pickup zone, hour and weather condition."""
import numpy as np
import pandas as pd


def weather_condition(scored: pd.DataFrame) -> pd.Series:
    cond = np.select(
        [scored["snowfall"].fillna(0) > 0, scored["precipitation"].fillna(0) > 0],
        ["snow", "rain"],
        default="clear",
    )
    return pd.Series(np.where(scored["temperature_2m"].isna() & (cond == "clear"), "unknown", cond),
                     index=scored.index)


def _summary(scored: pd.DataFrame, by, min_rows: int = 1) -> pd.DataFrame:
    grouped = scored.groupby(by, observed=True)
    table = pd.DataFrame({
        "rows": grouped.size(),
        "mae": grouped["abs_error"].mean(),
        "bias": grouped.apply(lambda g: (g["prediction"] - g["duration_min"]).mean(), include_groups=False),
    })
    return table[table["rows"] >= min_rows].sort_values("mae", ascending=False)


def error_analysis(scored: pd.DataFrame, min_zone_rows: int = 100, top_zones: int = 20) -> dict[str, pd.DataFrame]:
    """scored: output of compare.predict_frame. Returns tables keyed by dimension."""
    scored = scored.assign(weather_condition=weather_condition(scored))
    return {
        "by_pickup_zone": _summary(scored, "PULocationID", min_zone_rows).head(top_zones),
        "by_hour": _summary(scored, "hour").sort_index(),
        "by_weather": _summary(scored, "weather_condition"),
    }


def to_jsonable(tables: dict[str, pd.DataFrame]) -> dict:
    return {name: table.reset_index().to_dict(orient="records") for name, table in tables.items()}
