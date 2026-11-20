"""finish_month: DVC add -> repro -> push first, state file only after all three succeed. DVC is always mocked."""
import json
import subprocess

import pytest
from flows import month_state
from flows.month_state import DVC_STEPS, DvcError, finish_month, load_state, run_dvc, save_state


class FakeDvc:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, args):
        self.calls.append(tuple(args))
        if tuple(args) == self.fail_on:
            raise DvcError(f"`dvc {' '.join(args)}` failed")


def test_steps_are_add_raw_then_repro_then_push():
    assert DVC_STEPS == (("add", "data/raw"), ("repro",), ("push",))


def test_success_runs_the_steps_in_order_then_records_the_month(tmp_path):
    state, dvc = tmp_path / "state.json", FakeDvc()
    assert finish_month("2026-04", state, dvc) == ["2026-04"]
    assert dvc.calls == list(DVC_STEPS)
    data = json.loads(state.read_text())
    assert data["months"] == ["2026-04"] and data["updated_at"]
    assert finish_month("2026-05", state, dvc) == ["2026-04", "2026-05"] and load_state(state) == ["2026-04", "2026-05"]


def test_repeating_a_month_is_idempotent(tmp_path):
    state = tmp_path / "state.json"
    finish_month("2026-04", state, FakeDvc())
    assert finish_month("2026-04", state, FakeDvc()) == ["2026-04"]


@pytest.mark.parametrize("failing", list(DVC_STEPS))
def test_a_failing_dvc_step_leaves_the_state_file_untouched(tmp_path, failing):
    state = tmp_path / "state.json"
    save_state(["2026-03"], state)
    before = state.read_bytes()
    dvc = FakeDvc(fail_on=failing)
    with pytest.raises(DvcError):
        finish_month("2026-04", state, dvc)
    assert state.read_bytes() == before and load_state(state) == ["2026-03"]
    assert dvc.calls[-1] == failing and len(dvc.calls) == DVC_STEPS.index(failing) + 1  # later steps never ran
    assert not list(tmp_path.glob("*.tmp"))
    assert finish_month("2026-04", state, FakeDvc()) == ["2026-03", "2026-04"]  # a rerun repeats it and succeeds


def test_failing_before_any_state_exists_creates_no_file(tmp_path):
    state = tmp_path / "state.json"
    with pytest.raises(DvcError):
        finish_month("2026-04", state, FakeDvc(fail_on=DVC_STEPS[0]))
    assert not state.exists()


def test_a_corrupt_state_file_stops_before_any_dvc_work(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{not json")
    dvc = FakeDvc()
    with pytest.raises(ValueError):
        finish_month("2026-04", state, dvc)
    assert dvc.calls == [] and state.read_text() == "{not json"


def test_a_crash_while_writing_leaves_the_old_state_intact(tmp_path, monkeypatch):
    state = tmp_path / "state.json"
    save_state(["2026-03"], state)
    monkeypatch.setattr(month_state.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        finish_month("2026-04", state, FakeDvc())
    assert load_state(state) == ["2026-03"] and not list(tmp_path.glob("*.tmp"))


def test_missing_state_means_nothing_processed(tmp_path):
    assert load_state(tmp_path / "nope.json") == []


# ---- the real runner, with subprocess mocked -------------------------------------------------------------------------

def test_run_dvc_calls_python_dash_m_dvc_and_raises_with_the_output(monkeypatch, tmp_path):
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ERROR: remote unreachable")

    monkeypatch.setattr(month_state.subprocess, "run", fake_run)
    with pytest.raises(DvcError, match="remote unreachable"):
        run_dvc(["push"], cwd=tmp_path)
    cmd, kwargs = seen[0]
    assert cmd[1:] == ["-m", "dvc", "push"] and kwargs["cwd"] == tmp_path and kwargs["check"] is False


def test_run_dvc_missing_executable_is_a_dvc_error(monkeypatch):
    monkeypatch.setattr(month_state.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x")))
    with pytest.raises(DvcError):
        run_dvc(["repro"])
