"""dvc.yaml is wired to what the cleaning code really reads and writes (no DVC install needed)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STAGE = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]["clean"]


def test_clean_stage_deps_outs_and_metric_match_the_cleaning_module():
    assert STAGE["cmd"].startswith("python -m validation.clean")
    assert set(STAGE["deps"]) == {"data/raw", "src/validation/clean.py", "src/validation/schema.py"}
    assert all((ROOT / d).exists() for d in STAGE["deps"] if d != "data/raw")  # raw is DVC-tracked, may be absent
    assert STAGE["outs"] == ["data/processed/trips_clean.parquet"]
    assert STAGE["metrics"] == [{"data/processed/data_quality_report.json": {"cache": False}}]
    source = (ROOT / "src" / "validation" / "clean.py").read_text()
    assert "trips_clean.parquet" in source and "data_quality_report.json" in source  # the names the code writes


def test_raw_data_is_tracked_by_dvc_not_git():
    assert (ROOT / "data" / "raw.dvc").exists()
    assert "/data/raw/" in (ROOT / ".gitignore").read_text()
