"""Point-in-time rolling zone-pair duration features, vectorised.

Same semantics as the Feast snapshots (features.store.snapshots), verified by a test:
a trip counts for a query at time t only if it had COMPLETED by the hour boundary
H = floor(t): 1h = completions in (H-1h, H], 24h = completions in (H-24h, H].
Training and evaluation compute these from history; serving reads them from the Feast online store.
"""
import numpy as np
import pandas as pd

ROLLING_COLUMNS = ["avg_duration_1h", "avg_duration_24h", "trip_count_24h"]


def _pair(df: pd.DataFrame) -> np.ndarray:
    return (df["PULocationID"].astype("int64") * 1000 + df["DOLocationID"].astype("int64")).to_numpy()


def rolling_pair_features(history: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """history: cleaned trips (pickup_ts, PU, DO, duration_min). queries: rows with pickup_ts, PU, DO.

    Returns ROLLING_COLUMNS indexed like ``queries``. Pairs with no completions get NaN averages
    and trip_count_24h = 0.
    """
    end = pd.to_datetime(history["pickup_ts"]) + pd.to_timedelta(history["duration_min"], unit="m")
    done = pd.DataFrame({"pair": _pair(history), "bucket": end.dt.ceil("h").to_numpy(),
                         "dur": history["duration_min"].to_numpy()})
    agg = done.groupby(["pair", "bucket"], sort=False)["dur"].agg(["sum", "count"]).reset_index()
    agg = agg.sort_values(["bucket", "pair"]).reset_index(drop=True)
    agg["cum_sum"] = agg.groupby("pair")["sum"].cumsum()
    agg["cum_cnt"] = agg.groupby("pair")["count"].cumsum()

    q = pd.DataFrame({"pair": _pair(queries), "H": pd.to_datetime(queries["pickup_ts"]).dt.floor("h").to_numpy()})
    q["row"] = np.arange(len(q))
    q = q.sort_values("H").reset_index(drop=True)

    def cumulative_at(shift_hours: int) -> np.ndarray:
        left = q.assign(T=q["H"] - pd.Timedelta(hours=shift_hours)).sort_values("T")
        merged = pd.merge_asof(left, agg[["pair", "bucket", "cum_sum", "cum_cnt"]], left_on="T",
                               right_on="bucket", by="pair", direction="backward")
        return merged.sort_values("row")[["cum_sum", "cum_cnt"]].fillna(0.0).to_numpy()

    now, before = cumulative_at(0), cumulative_at(24)
    sum24, cnt24 = now[:, 0] - before[:, 0], now[:, 1] - before[:, 1]

    exact = q[["row", "pair", "H"]].merge(agg[["pair", "bucket", "sum", "count"]], how="left",
                                          left_on=["pair", "H"], right_on=["pair", "bucket"]).sort_values("row")
    sum1, cnt1 = exact["sum"].to_numpy(), exact["count"].to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({
            "avg_duration_1h": np.where(cnt1 > 0, sum1 / cnt1, np.nan).astype("float32"),
            "avg_duration_24h": np.where(cnt24 > 0, sum24 / cnt24, np.nan).astype("float32"),
            "trip_count_24h": cnt24.astype("float32"),
        }, index=queries.index)
