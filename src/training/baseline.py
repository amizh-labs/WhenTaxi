import pandas as pd


class ZonePairMeanBaseline:
    """Predict the mean duration of the (pickup zone, dropoff zone) pair.

    Unseen pairs fall back to the pickup-zone mean, then to the global mean.
    """

    def fit(self, trips: pd.DataFrame) -> "ZonePairMeanBaseline":
        keys = ["PULocationID", "DOLocationID"]
        self.pair_mean_ = trips.groupby(keys)["duration_min"].mean()
        self.pu_mean_ = trips.groupby("PULocationID")["duration_min"].mean()
        self.global_mean_ = float(trips["duration_min"].mean())
        return self

    def predict(self, trips: pd.DataFrame) -> pd.Series:
        index = pd.MultiIndex.from_frame(trips[["PULocationID", "DOLocationID"]])
        pair = pd.Series(self.pair_mean_.reindex(index).to_numpy(), index=trips.index)
        pu = trips["PULocationID"].map(self.pu_mean_)
        return pair.fillna(pu).fillna(self.global_mean_)
