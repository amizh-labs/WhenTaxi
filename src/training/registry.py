"""Load a registered model together with the zone categories it was trained with."""
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from features import GEO_COLUMNS, ROLLING_COLUMNS, ZONE_COLUMNS, apply_zone_categories
from training.train import MODEL_NAME, default_tracking_uri


@dataclass
class LoadedModel:
    model: object
    zone_categories: list[int]
    version: str
    log_target: bool = False  # trained on log1p(duration): predictions are expm1'd

    @property
    def feature_columns(self) -> list[str]:
        """Exactly the columns (and order) the model was trained on."""
        return list(self.model.feature_name_)

    @property
    def needs_rolling(self) -> bool:
        return set(ROLLING_COLUMNS) <= set(self.feature_columns)

    @property
    def needs_geo(self) -> bool:
        return set(GEO_COLUMNS) <= set(self.feature_columns)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """features: output of build_features (with rolling columns when the model needs them)."""
        frame = features[self.feature_columns].copy()
        for col in ZONE_COLUMNS:
            frame[col] = apply_zone_categories(frame[col].astype(object), self.zone_categories)
        raw = np.asarray(self.model.predict(frame))
        return pd.Series(np.expm1(raw) if self.log_target else raw, index=features.index)


def load_model(alias: str = "champion", tracking_uri: str | None = None, name: str = MODEL_NAME) -> LoadedModel:
    uri = tracking_uri or default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    version = MlflowClient(uri).get_model_version_by_alias(name, alias)
    model = mlflow.lightgbm.load_model(f"models:/{name}@{alias}")
    categories = mlflow.artifacts.load_dict(f"runs:/{version.run_id}/zone_categories.json")
    params = MlflowClient(uri).get_run(version.run_id).data.params
    return LoadedModel(model=model, zone_categories=[int(z) for z in categories], version=str(version.version),
                       log_target=params.get("log_target") == "True")
