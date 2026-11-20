"""Pandera schema for the cleaned table (docs/contracts.md section 1)."""
import pandera.pandas as pa
from pandera.typing import Series


class CleanTripsSchema(pa.DataFrameModel):
    pickup_ts: Series[pa.DateTime]
    PULocationID: Series[int] = pa.Field(ge=1)
    DOLocationID: Series[int] = pa.Field(ge=1)
    duration_min: Series[float] = pa.Field(ge=1.0, le=180.0)
    month: Series[str] = pa.Field(str_matches=r"^\d{4}-\d{2}$")

    class Config:
        strict = True
        coerce = False
