"""Retrain trigger logic (pure functions, no Prefect needed to test).

Retrain when EITHER a new month of cleaned data exists that the current model has not
been trained/validated on, OR the latest drift report says the dataset drifted.
Drift report format: docs/contracts.md section 5 (drift_report.json).
"""
import json
from pathlib import Path


def read_drift(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text())


def should_retrain(available_months: list[str], used_months: list[str], drift: dict | None) -> tuple[bool, str]:
    """Return (decision, reason). used_months = train+valid months of the champion's run."""
    new = sorted(set(available_months) - set(used_months))
    if new:
        return True, f"new data: {', '.join(new)}"
    if drift and drift.get("dataset_drift"):
        share = drift.get("share_drifted_features")
        return True, f"drift detected (share drifted features: {share})"
    return False, "no new data and no drift"
