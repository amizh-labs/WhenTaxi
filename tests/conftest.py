"""Test-suite setup.

Importing feast sets PROMETHEUS_MULTIPROC_DIR; prometheus_client picks single- vs multi-process
value storage when it is first imported. Import it here, before any test module can pull in
feast, so the serving metrics tests keep normal per-app counters.
"""
import prometheus_client  # noqa: F401
import pytest


class NetworkBlocked(BaseException):  # BaseException: the live-weather fallback must not be able to swallow it
    pass


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """No test may call Open-Meteo: live weather is off by default and the real HTTP function raises if reached."""
    monkeypatch.setenv("ETA_LIVE_WEATHER", "0")

    def blocked(*args, **kwargs):
        raise NetworkBlocked("a test tried to call the real network")

    try:
        import serving.live_weather as live
    except ImportError:  # serving extra not installed
        return
    monkeypatch.setattr(live, "http_get_json", blocked)
