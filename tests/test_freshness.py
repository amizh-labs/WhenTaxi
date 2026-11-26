"""Stage L3: feature freshness. Metadata file -> eta_feature_age_seconds -> alert rule -> compose / cluster / Grafana."""
import json
import re
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from features import feature_age_seconds, write_materialization_metadata
from features.freshness import FILENAME, MISSING_AGE
from serving import create_app

ROOT = Path(__file__).resolve().parents[1]
RULES = yaml.safe_load((ROOT / "monitoring" / "alerts.yml").read_text())
RULE = RULES["groups"][0]["rules"][0]


class StubModel:
    version = "7"
    zone_categories = tuple(range(1, 266))

    def predict(self, features):
        return features["hour"] * 0.0 + 10.0


def sample(client, name):
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[-1])
    raise AssertionError(f"{name} not exported")


@pytest.fixture
def feast_root(tmp_path, monkeypatch):
    root = tmp_path / "feast"
    root.mkdir()
    monkeypatch.setenv("ETA_FEAST_ROOT", str(root))
    return root


def make_client(tmp_path):
    return TestClient(create_app(StubModel, tmp_path / "none.json", rolling_provider=None, live_weather=None,
                                 prediction_log=None))


def evaluate(expr: str, client) -> bool:
    """Evaluate the alert expression the way Prometheus would for this one pod: `<metric> > <metric>`."""
    left, op, right = re.fullmatch(r"\s*(\w+)\s*(>)\s*(\w+)\s*", expr).groups()
    assert op == ">"
    return sample(client, left) > sample(client, right)


# ---- the metadata file and the gauge ---------------------------------------------------------------------------

def test_write_and_read_metadata(tmp_path):
    now = time.time()
    path = write_materialization_metadata(tmp_path / "store", upto=pd.Timestamp("2026-03-31 23:00"), now=now - 600)
    assert path.name == FILENAME and json.loads(path.read_text())["upto"] == "2026-03-31 23:00:00"
    assert feature_age_seconds(tmp_path / "store", now=now) == pytest.approx(600, abs=1)
    assert feature_age_seconds(tmp_path / "store", now=now - 5000) == 0.0  # clock behind the file: "just now", not negative


@pytest.mark.parametrize("content", [None, "", "not json", "{}", '{"materialized_at": "yesterday"}', "[]"])
def test_missing_or_broken_metadata_means_unknown_not_a_crash(tmp_path, content):
    root = tmp_path / "store"
    root.mkdir()
    if content is not None:
        (root / FILENAME).write_text(content)
    assert feature_age_seconds(root) is None
    assert feature_age_seconds(tmp_path / "does-not-exist") is None


def test_gauge_reflects_the_metadata_file(tmp_path, feast_root):
    write_materialization_metadata(feast_root, now=time.time() - 3600)
    with make_client(tmp_path) as client:
        assert sample(client, "eta_feature_age_seconds") == pytest.approx(3600, abs=5)
        write_materialization_metadata(feast_root, now=time.time() - 10)  # re-materialized: the next scrape is fresh
        assert sample(client, "eta_feature_age_seconds") == pytest.approx(10, abs=5)


def test_missing_metadata_gauge_is_the_safe_value_and_the_api_keeps_serving(tmp_path, feast_root):
    with make_client(tmp_path) as client:
        assert sample(client, "eta_feature_age_seconds") == MISSING_AGE == -1.0
        resp = client.post("/predict", json={"pickup_zone": 1, "dropoff_zone": 2, "pickup_ts": "2026-03-05T08:30:00"})
        assert resp.status_code == 200
        (feast_root / FILENAME).write_text("{broken")
        assert sample(client, "eta_feature_age_seconds") == -1.0


def test_materialize_writes_the_metadata_file(tmp_path):
    from features.store import build_online_store

    trips = pd.read_parquet(ROOT / "tests" / "fixtures" / "trips_clean_sample.parquet")
    root = tmp_path / "feast"
    build_online_store(trips, root, trips["pickup_ts"].max().ceil("h"))
    assert feature_age_seconds(root) == pytest.approx(0, abs=60)


# ---- the alert expression ---------------------------------------------------------------------------------------

def test_alert_rule_shape():
    assert RULE["alert"] == "EtaFeaturesStale"
    assert RULE["expr"] == "eta_feature_age_seconds > eta_feature_age_alert_threshold_seconds"
    assert RULE["for"] and RULE["annotations"]["summary"]


UNKNOWN_RULE = next(r for r in RULES["groups"][0]["rules"] if r["alert"] == "EtaFeaturesUnknown")


def test_unknown_alert_rule_shape_and_the_two_rules_never_overlap():
    assert UNKNOWN_RULE["expr"] == "eta_feature_age_seconds < 0"
    assert UNKNOWN_RULE["for"] and UNKNOWN_RULE["annotations"]["summary"]
    assert [r["alert"] for r in RULES["groups"][0]["rules"]] == ["EtaFeaturesStale", "EtaFeaturesUnknown"]


def test_unknown_age_fires_only_the_unknown_alert_and_a_known_age_fires_neither(tmp_path, feast_root):
    with make_client(tmp_path) as client:  # no materialized.json: age is -1
        age, threshold = sample(client, "eta_feature_age_seconds"), sample(client, "eta_feature_age_alert_threshold_seconds")
        assert age < 0 and not age > threshold  # Unknown fires, Stale stays quiet
        write_materialization_metadata(feast_root, now=time.time() - 60)
        age = sample(client, "eta_feature_age_seconds")
        assert not age < 0 and not age > threshold  # fresh: neither fires


def test_alert_fires_when_the_threshold_is_below_the_current_age(tmp_path, feast_root, monkeypatch):
    write_materialization_metadata(feast_root, now=time.time() - 3600)  # one hour old
    monkeypatch.setenv("ETA_FEATURE_AGE_ALERT_SECONDS", "60")  # threshold well below the age
    with make_client(tmp_path) as client:
        assert sample(client, "eta_feature_age_alert_threshold_seconds") == 60.0
        assert evaluate(RULE["expr"], client) is True


def test_alert_stays_quiet_below_threshold_with_default_threshold_and_when_unknown(tmp_path, feast_root, monkeypatch):
    write_materialization_metadata(feast_root, now=time.time() - 3600)
    monkeypatch.delenv("ETA_FEATURE_AGE_ALERT_SECONDS", raising=False)
    with make_client(tmp_path) as client:
        assert sample(client, "eta_feature_age_alert_threshold_seconds") == 7 * 24 * 3600
        assert evaluate(RULE["expr"], client) is False
        (feast_root / FILENAME).unlink()  # age unknown (-1): never fires
        assert evaluate(RULE["expr"], client) is False


# ---- wiring: compose, cluster, Grafana, and no prediction log in the cluster ----------------------------------------

def test_compose_loads_the_rules_and_sets_the_log_and_threshold():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert "./monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro" in compose["services"]["prometheus"]["volumes"]
    assert "/etc/prometheus/alerts.yml" in yaml.safe_load((ROOT / "monitoring" / "prometheus.yml").read_text())["rule_files"]
    env = compose["services"]["api"]["environment"]
    assert env["ETA_PREDICTION_LOG"] == "/app/log/predictions.sqlite" and "ETA_FEATURE_AGE_ALERT_SECONDS" in env
    assert "prediction-log:/app/log" in compose["services"]["api"]["volumes"] and "prediction-log" in compose["volumes"]


def test_cluster_prometheus_carries_the_same_rules_and_the_api_does_not_log_predictions():
    docs = list(yaml.safe_load_all((ROOT / "k8s" / "prometheus.yaml").read_text()))
    config = next(d for d in docs if d["kind"] == "ConfigMap")["data"]
    assert yaml.safe_load(config["alerts.yml"]) == RULES  # identical to monitoring/alerts.yml
    assert yaml.safe_load(config["prometheus.yml"])["rule_files"] == ["/etc/prometheus/alerts.yml"]
    rollout = (ROOT / "k8s" / "rollout.yaml").read_text()
    assert "ETA_PREDICTION_LOG" not in rollout  # 5-10 replicas would fight over one SQLite file
    assert "ETA_FEATURE_AGE_ALERT_SECONDS" in rollout


def test_grafana_dashboard_has_a_feature_age_panel():
    dashboard = json.loads((ROOT / "monitoring" / "grafana" / "dashboards" / "eta.json").read_text())
    panel = next(p for p in dashboard["panels"] if "Feature age" in p["title"])
    exprs = " ".join(t["expr"] for t in panel["targets"])
    assert "eta_feature_age_seconds" in exprs and "eta_feature_age_alert_threshold_seconds" in exprs
    assert len({p["id"] for p in dashboard["panels"]}) == len(dashboard["panels"])  # ids stay unique


def test_grafana_dashboard_panels_use_known_metrics_and_do_not_overlap():
    dashboard = json.loads((ROOT / "monitoring" / "grafana" / "dashboards" / "eta.json").read_text())
    known = {"eta_requests_total", "eta_request_latency_seconds_bucket", "eta_prediction_minutes_bucket",
             "eta_prediction_minutes_sum", "eta_prediction_minutes_count", "eta_feature_age_seconds",
             "eta_feature_age_alert_threshold_seconds", "ALERTS"}
    cells = set()
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            names = set(re.findall(r"\b(?:eta_[a-z_]+|ALERTS)\b", target["expr"]))
            assert names and names <= known, (panel["title"], names - known)
        g = panel["gridPos"]
        panel_cells = {(x, y) for x in range(g["x"], g["x"] + g["w"]) for y in range(g["y"], g["y"] + g["h"])}
        assert not panel_cells & cells, panel["title"]  # no two panels share a grid cell
        cells |= panel_cells
