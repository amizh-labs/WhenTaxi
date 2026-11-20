import pandas as pd
import pandera.errors
import pytest

from validation import CleanTripsSchema, clean_trips

VALID = set(range(1, 266))
UNKNOWN = {264, 265}


def raw(rows):
    """rows: (pickup, dropoff, PU, DO)"""
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime([r[0] for r in rows], format="mixed"),
            "tpep_dropoff_datetime": pd.to_datetime([r[1] for r in rows], format="mixed"),
            "PULocationID": [r[2] for r in rows],
            "DOLocationID": [r[3] for r in rows],
        }
    )


GOOD = ("2026-03-05 08:00", "2026-03-05 08:20", 10, 20)


def run(*rows):
    return clean_trips(raw(list(rows)), "2026-03", VALID, UNKNOWN)


def test_good_row_kept_with_contract_columns():
    clean, report = run(GOOD)
    assert list(clean.columns) == ["pickup_ts", "PULocationID", "DOLocationID", "duration_min", "month"]
    assert clean["duration_min"].iloc[0] == 20.0
    assert clean["month"].iloc[0] == "2026-03"
    assert report["rows_out"] == 1


def test_pickup_outside_month_dropped_even_2008():
    _, report = run(GOOD, ("2008-12-31 23:00", "2008-12-31 23:20", 10, 20),
                    ("2026-02-28 23:50", "2026-03-01 00:10", 10, 20))
    assert report["dropped_outside_month"] == 2 and report["rows_out"] == 1


@pytest.mark.parametrize("dropoff", ["2026-03-05 08:00", "2026-03-05 07:00", "2026-03-05 08:00:30", "2026-03-05 11:01"])
def test_bad_durations_dropped(dropoff):
    _, report = run(GOOD, ("2026-03-05 08:00", dropoff, 10, 20))
    assert report["dropped_bad_duration"] == 1


def test_duration_bounds_inclusive():
    clean, _ = run(("2026-03-05 08:00", "2026-03-05 08:01", 10, 20),
                   ("2026-03-05 08:00", "2026-03-05 11:00", 10, 20))
    assert len(clean) == 2


def test_zone_not_in_lookup_and_unknown_zone_are_separate_rules():
    _, report = run(GOOD, ("2026-03-05 08:00", "2026-03-05 08:20", 999, 20),
                    ("2026-03-05 08:00", "2026-03-05 08:20", 10, 264),
                    ("2026-03-05 08:00", "2026-03-05 08:20", 265, 20))
    assert report["dropped_zone_not_in_lookup"] == 1
    assert report["dropped_unknown_or_outside_zone"] == 2


def test_each_row_counted_once_and_totals_add_up():
    # breaks month AND duration AND zone: attributed to the first rule only
    _, report = run(GOOD, ("2008-01-01 00:00", "2008-01-01 00:00", 999, 20))
    assert report["dropped_outside_month"] == 1
    assert report["dropped_bad_duration"] == 0 and report["dropped_zone_not_in_lookup"] == 0
    dropped = sum(v for k, v in report.items() if k.startswith("dropped_"))
    assert report["rows_in"] == dropped + report["rows_out"]


def test_schema_rejects_bad_table():
    clean, _ = run(GOOD)
    errors = (pandera.errors.SchemaError, pandera.errors.SchemaErrors)
    with pytest.raises(errors):
        CleanTripsSchema.validate(clean.assign(duration_min=500.0))
    with pytest.raises(errors):
        CleanTripsSchema.validate(clean.assign(extra=1))
