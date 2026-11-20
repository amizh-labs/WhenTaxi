"""scripts/verify.py end to end on the small fixture (stretched over Jan-Mar so a time split exists)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "trips_clean_sample.parquet"
VERIFY = Path("scripts") / "verify.py"
NAMES = ["leakage", "time-split", "weather", "baseline", "seed", "unseen-zone"]


@pytest.fixture(scope="module")
def verify():
    spec = importlib.util.spec_from_file_location("verify_script", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def three_months(tmp_path_factory):
    """The 500-row January fixture split in thirds and shifted into Jan / Feb / Mar."""
    trips = pd.read_parquet(FIXTURE)
    third = len(trips) // 3
    parts = []
    for i, (month, shift) in enumerate((("2026-01", 0), ("2026-02", 31), ("2026-03", 59))):
        part = trips.iloc[i * third:(i + 1) * third if i < 2 else len(trips)].copy()
        part["pickup_ts"] += pd.Timedelta(days=shift)
        part["month"] = month
        parts.append(part)
    path = tmp_path_factory.mktemp("verify") / "trips.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


def run(verify, trips, capsys, *extra):
    code = verify.main(["--trips", str(trips), "--weather", "none.json", "--zones", "none.csv", *extra])
    return code, capsys.readouterr().out


def test_all_six_checks_pass_on_the_fixture_with_numbers(verify, three_months, capsys):
    code, out = run(verify, three_months, capsys)
    lines = [line for line in out.splitlines() if line.startswith(("PASS", "FAIL"))]
    assert code == 0, out
    assert [line.split()[1] for line in lines] == NAMES and all(line.startswith("PASS") for line in lines)
    assert "6/6 checks passed" in out
    assert "tip_amount" in out and "MAE" in out and "identical" in out  # numbers and the forbidden list are shown


def test_exit_code_is_1_when_a_check_fails(verify, three_months, capsys):
    # validation months BEFORE the training months: the time-split check must catch it
    code, out = run(verify, three_months, capsys, "--train-months", "2026-03", "--valid-months", "2026-01")
    assert code == 1
    assert next(line for line in out.splitlines() if " time-split " in line).startswith("FAIL")
    assert "checks passed" in out and "6/6" not in out


def test_leakage_check_covers_tip_amount(verify):
    assert set(verify.FORBIDDEN) == {"trip_distance", "fare_amount", "total_amount", "tip_amount"}
