import importlib.util
import socket
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1]
ROOT = UI.parent


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """The UI tests must never touch a real network: any connection attempt fails loudly."""
    def refuse(*args, **kwargs):
        raise AssertionError("a UI test tried to open a real network connection")

    # urllib3 (and so requests) opens connections through create_connection; asyncio's own loopback socketpair on
    # Windows uses socket.connect, which must keep working, so that one is deliberately not patched.
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr("requests.adapters.HTTPAdapter.send", refuse)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ui_app():
    return load_module("ui_app", UI / "app.py")


@pytest.fixture(scope="session")
def build_zones():
    return load_module("build_ui_zones", ROOT / "scripts" / "build_ui_zones.py")


def pytest_sessionfinish(session, exitstatus):
    """A skipped UI test is a failure: the CI job must really run every one of them."""
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None and reporter.stats.get("skipped"):
        session.exitstatus = 1
