"""CI data checks: the committed sample must satisfy the cleaned-table contract."""
import pandas as pd

from validation import CleanTripsSchema

SAMPLE = "tests/fixtures/trips_clean_sample.parquet"


def test_fixture_satisfies_pandera_schema():
    CleanTripsSchema.validate(pd.read_parquet(SAMPLE))


def test_fixture_has_contract_columns_and_no_leakage_columns():
    frame = pd.read_parquet(SAMPLE)
    assert list(frame.columns) == ["pickup_ts", "PULocationID", "DOLocationID", "duration_min", "month"]
    assert not {"trip_distance", "fare_amount", "total_amount"} & set(frame.columns)
