"""Unit tests for the gate mechanism, using the fast synthetic dataset.

The real-data gate (python -m evaluation.gate) runs in CI against real NYC taxi data.
"""
import json

import pytest

from evaluation import gate
from evaluation.gate import REFERENCE_REAL, REFERENCE_SYNTHETIC, check, gate_metrics, synthetic_trips


def test_check_thresholds():
    ref = {"mae": 4.0}
    assert check({"mae": 4.0}, ref)[0] is True
    assert check({"mae": 3.5}, ref)[0] is True  # improvements always pass
    assert check({"mae": 4.07}, ref)[0] is True  # +1.75%
    ok, message = check({"mae": 4.2}, ref)  # +5%
    assert not ok and message.startswith("FAIL")


def test_synthetic_data_is_deterministic():
    assert synthetic_trips().equals(synthetic_trips())


def test_current_code_passes_the_synthetic_reference():
    reference = json.loads(REFERENCE_SYNTHETIC.read_text())
    ok, message = check(gate_metrics(), reference)
    assert ok, message


def test_gate_catches_a_degraded_model(monkeypatch):
    monkeypatch.setattr(gate, "GATE_PARAMS", {**gate.GATE_PARAMS, "n_estimators": 2, "num_leaves": 2})
    reference = json.loads(REFERENCE_SYNTHETIC.read_text())
    ok, message = check(gate.gate_metrics(), reference)
    assert not ok, message


@pytest.mark.parametrize("path", [REFERENCE_SYNTHETIC, REFERENCE_REAL])
@pytest.mark.parametrize("key", ["mae", "rmse", "p90_abs_error"])
def test_reference_files_have_expected_keys(path, key):
    assert key in json.loads(path.read_text())
