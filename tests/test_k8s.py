"""Static checks for the k8s manifests and demo helpers (no cluster needed)."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

K8S = Path("k8s")


def load(name: str) -> list[dict]:
    return [doc for doc in yaml.safe_load_all((K8S / name).read_text()) if doc]


def by_kind(docs: list[dict], kind: str) -> dict:
    return next(d for d in docs if d["kind"] == kind)


def rollout_doc() -> dict:
    return by_kind(load("rollout.yaml"), "Rollout")


def pod_spec() -> dict:
    return rollout_doc()["spec"]["template"]["spec"]


def template_metrics() -> list[dict]:
    return by_kind(load("analysis-template.yaml"), "AnalysisTemplate")["spec"]["metrics"]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("k8s_demo", K8S / "demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["mlflow.yaml", "prometheus.yaml", "analysis-template.yaml", "rollout.yaml",
                                  "traffic.yaml"])
def test_manifests_parse_and_use_eta_namespace(name):
    for doc in load(name):
        assert doc["apiVersion"] and doc["kind"]
        if doc["kind"] not in ("ClusterRole", "ClusterRoleBinding"):
            assert doc["metadata"]["namespace"] == "eta"


# ---- 1. hash label and args --------------------------------------------------------------

def test_canary_is_10_50_100_with_pauses_and_background_analysis():
    canary = rollout_doc()["spec"]["strategy"]["canary"]
    assert [s["setWeight"] for s in canary["steps"] if "setWeight" in s] == [10, 50, 100]
    assert any("pause" in s for s in canary["steps"])
    assert canary["analysis"]["templates"] == [{"templateName": "eta-canary-health"}]


def test_hash_args_use_podTemplateHashValue_latest_and_stable():
    args = rollout_doc()["spec"]["strategy"]["canary"]["analysis"]["args"]
    assert {a["name"]: a["valueFrom"] for a in args} == {
        "canary-hash": {"podTemplateHashValue": "Latest"},
        "stable-hash": {"podTemplateHashValue": "Stable"},
    }


def test_prometheus_relabels_the_rollouts_hash_label_not_deployment_style():
    config = yaml.safe_load(by_kind(load("prometheus.yaml"), "ConfigMap")["data"]["prometheus.yml"])
    relabels = config["scrape_configs"][0]["relabel_configs"]
    hash_rule = next(r for r in relabels if r.get("target_label") == "rollouts_pod_template_hash")
    assert hash_rule["source_labels"] == ["__meta_kubernetes_pod_label_rollouts_pod_template_hash"]
    # a bare Deployment-style `pod_template_hash` label would never be set on Rollout pods
    assert not re.search(r"pod_label_pod_template_hash", json.dumps(relabels))


def test_queries_filter_on_the_same_label_prometheus_produces():
    for metric in template_metrics():
        query = metric["provider"]["prometheus"]["query"]
        assert "rollouts_pod_template_hash=" in query
        assert re.search(r'(?<!rollouts_)pod_template_hash', query) is None


def test_template_args_are_all_declared_and_used():
    template = by_kind(load("analysis-template.yaml"), "AnalysisTemplate")["spec"]
    declared = {a["name"] for a in template["args"]}
    used = set(re.findall(r"\{\{args\.([\w-]+)\}\}", json.dumps(template["metrics"])))
    assert used == declared == {"canary-hash", "stable-hash"}


def test_analysis_template_uses_contract_metric_names():
    """docs/contracts.md section 4: the API exports these, the canary analysis must query them."""
    queries = " ".join(m["provider"]["prometheus"]["query"] for m in template_metrics())
    for name in ("eta_request_latency_seconds_bucket", "eta_requests_total", "eta_prediction_minutes_sum",
                 "eta_prediction_minutes_count"):
        assert name in queries
    assert {m["name"] for m in template_metrics()} == {"p95-latency-seconds", "error-rate-5xx",
                                                       "prediction-sanity-vs-stable"}


# ---- 2. no data is never a pass ----------------------------------------------------------

@pytest.mark.parametrize("metric", template_metrics(), ids=lambda m: m["name"])
def test_no_data_or_nan_is_never_a_pass_and_is_inconclusive(metric):
    success, failure = metric["successCondition"], metric["failureCondition"]
    # success must positively require data that is not NaN, so empty/NaN can never pass ...
    assert "len(result) > 0" in success and "!isNaN(result[0])" in success
    # ... a failure needs the same guard, so empty/NaN is neither success nor failure = Inconclusive
    assert "len(result) > 0" in failure and "!isNaN(result[0])" in failure
    assert metric["inconclusiveLimit"] == 3


@pytest.mark.parametrize("metric", template_metrics(), ids=lambda m: m["name"])
def test_real_breach_aborts_immediately_and_errors_are_bounded(metric):
    assert metric["failureLimit"] == 0  # the first real threshold breach rolls back
    assert metric["consecutiveErrorLimit"] <= 2  # persistent query/eval errors also end in a failed run


@pytest.mark.parametrize("metric", template_metrics(), ids=lambda m: m["name"])
def test_initial_delay_covers_pod_start_scrape_and_a_rate_window(metric):
    # the canary must load the model, be scraped (5s) and get traffic before rate(...[1m]) has samples
    assert int(metric["initialDelay"].rstrip("s")) >= 120


@pytest.mark.parametrize("metric", template_metrics(), ids=lambda m: m["name"])
def test_conditions_are_complements_on_a_present_value(metric):
    threshold = re.search(r"result\[0\] < ([\d.]+)", metric["successCondition"]).group(1)
    assert f"result[0] >= {threshold}" in metric["failureCondition"]


# ---- 3. Feast store is never mounted read-only from the host ------------------------------

def test_feast_store_is_copied_into_a_writable_emptydir():
    spec = pod_spec()
    volumes = {v["name"]: v for v in spec["volumes"]}
    assert "emptyDir" in volumes["feast"]

    init = next(c for c in spec["initContainers"] if c["name"] == "copy-feast-store")
    mounts = {m["name"]: m for m in init["volumeMounts"]}
    assert mounts["feast-src"].get("readOnly") is True  # the host snapshot is only read ...
    assert "readOnly" not in mounts["feast"]  # ... and copied into a writable emptyDir
    assert "cp -r /src/." in " ".join(init["command"])

    api = next(c for c in spec["containers"] if c["name"] == "api")
    api_mount = next(m for m in api["volumeMounts"] if m["name"] == "feast")
    assert api_mount["mountPath"] == "/data/feast" and not api_mount.get("readOnly")
    assert not any(m["name"] == "feast-src" for m in api["volumeMounts"])  # API never touches host files


# ---- 4. replicas option and memory sizing --------------------------------------------------

def test_default_replicas_make_canary_weights_exact():
    assert rollout_doc()["spec"]["replicas"] == 10


def test_replicas_option_defaults_to_10_and_allows_5(demo):
    assert demo.DEFAULT_REPLICAS == 10 and demo.ALLOWED_REPLICAS == (5, 10)
    assert yaml.safe_load(demo.render_rollout(5).split("---")[0])["spec"]["replicas"] == 5
    assert demo.validate_replicas(10) == 10
    for bad in (0, 3, 7, 20):
        with pytest.raises(ValueError):
            demo.validate_replicas(bad)


def test_render_changes_only_replicas_and_keeps_the_service(demo):
    docs = list(yaml.safe_load_all(demo.render_rollout(5)))
    original = load("rollout.yaml")
    assert [d["kind"] for d in docs] == ["Rollout", "Service"]
    rendered = {**docs[0], "spec": {**docs[0]["spec"], "replicas": 10}}
    assert rendered == original[0] and docs[1] == original[1]


def test_api_pods_have_small_memory_requests_and_limits():
    api = next(c for c in pod_spec()["containers"] if c["name"] == "api")
    to_mi = lambda s: int(s.removesuffix("Mi"))
    request, limit = to_mi(api["resources"]["requests"]["memory"]), to_mi(api["resources"]["limits"]["memory"])
    # measured ~345 MB RSS: the limit must clear that, but stay well below a GB so 10 pods fit
    assert 345 < limit <= 512 and request <= limit and request <= 256


# ---- demo.py behaviour ---------------------------------------------------------------------

def test_commands_are_printed_exactly_as_run(demo, capsys):
    demo.sh(sys.executable, "-c", "print('hi')")
    out = capsys.readouterr().out
    assert out.startswith(f"$ {sys.executable} -c print('hi')") and "hi" in out


def test_quiet_polling_does_not_repeat_the_command(demo, capsys):
    demo.sh(sys.executable, "-c", "pass", quiet=True)
    assert "$" not in capsys.readouterr().out


def test_rollout_has_a_fault_switch_defaulting_to_off(demo):
    rollout = rollout_doc()
    env = {e["name"]: e["value"] for e in rollout["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["ETA_FAULT"] == ""  # a normal release is healthy; "bad" flips this to a faulty one
    patch = json.loads(demo.env_patch(rollout, "ETA_FAULT", "bias"))[0]
    assert patch["op"] == "replace" and patch["value"] == "bias"
    assert patch["path"].endswith(f"/env/{demo.env_index(rollout, 'ETA_FAULT')}/value")


def test_new_revision_detection_prevents_false_success(demo):
    stale = {"metadata": {"generation": 2}, "status": {"observedGeneration": "1", "currentPodHash": "old",
                                                        "phase": "Healthy"}}
    # patched (generation 2) but Argo has not observed it yet: still the old Healthy revision
    assert demo.revision_changed("old", stale) is False
    seen_but_same = {"metadata": {"generation": 2}, "status": {"observedGeneration": "2", "currentPodHash": "old"}}
    assert demo.revision_changed("old", seen_but_same) is False  # template unchanged -> nothing to test
    new = {"metadata": {"generation": 2}, "status": {"observedGeneration": "2", "currentPodHash": "new"}}
    assert demo.revision_changed("old", new) is True
    assert demo.revision_changed("old", {"metadata": {"generation": 1}}) is False


def test_good_and_bad_both_change_the_pod_template(demo):
    """Each release must alter the pod template, otherwise Argo creates no revision and tests nothing."""
    rollout = rollout_doc()
    base = json.dumps(rollout["spec"]["template"], sort_keys=True)
    for name, value in (("ETA_RELEASE", "1700000000"), ("ETA_FAULT", "bias")):
        patch = json.loads(demo.env_patch(rollout, name, value))[0]
        idx = demo.env_index(rollout, name)
        patched = json.loads(json.dumps(rollout))
        patched["spec"]["template"]["spec"]["containers"][0]["env"][idx]["value"] = patch["value"]
        assert json.dumps(patched["spec"]["template"], sort_keys=True) != base


def test_argo_install_is_applied_server_side(demo):
    """The Argo CRDs are larger than the client-side apply annotation limit (262144 bytes)."""
    assert "--server-side" in demo.ARGO_APPLY and "--force-conflicts" in demo.ARGO_APPLY
    assert demo.ARGO_APPLY[-1].endswith("install.yaml") and "argo-rollouts" in demo.ARGO_APPLY


def test_classify_rollout_states(demo):
    assert demo.classify({"status": {"phase": "Healthy"}}) == "healthy"
    assert demo.classify({"status": {"phase": "Degraded", "abort": True}}) == "aborted"
    assert demo.classify({"status": {"phase": "Progressing"}}) == "progressing"
    assert demo.classify({}) == "progressing"


def test_cluster_config_mounts_data_dirs(demo):
    config = yaml.safe_load(demo.cluster_config(Path("/repo")))
    volumes = [v["volume"] for v in config["volumes"]]
    assert volumes == ["/repo/data/raw:/data/raw", "/repo/data/feast:/data/feast"]
    assert config["metadata"]["name"] == "eta"


def test_no_stable_service_so_canary_pods_receive_traffic():
    """stableService makes Argo pin the Service to the stable ReplicaSet, starving the canary of traffic."""
    canary = rollout_doc()["spec"]["strategy"]["canary"]
    assert "stableService" not in canary and "canaryService" not in canary
    service = by_kind(load("rollout.yaml"), "Service")
    assert service["spec"]["selector"] == {"app": "eta-api"}


def test_mlflow_state_lives_on_a_local_path_pvc():
    docs = load("mlflow.yaml")
    claim = by_kind(docs, "PersistentVolumeClaim")
    assert claim["spec"]["storageClassName"] == "local-path"  # k3d's default provisioner
    assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]

    deploy = by_kind(docs, "Deployment")
    pod = deploy["spec"]["template"]["spec"]
    volume = next(v for v in pod["volumes"] if v["name"] == "data")
    assert volume == {"name": "data", "persistentVolumeClaim": {"claimName": claim["metadata"]["name"]}}
    assert "emptyDir" not in volume

    container = pod["containers"][0]
    mount = next(m for m in container["volumeMounts"] if m["name"] == "data")
    assert mount["mountPath"] == "/mlflow"
    # both the SQLite DB and the artifacts must be under the mounted path, or they would not persist
    command = " ".join(container["command"])
    assert "sqlite:////mlflow/mlflow.db" in command and "/mlflow/artifacts" in command
    assert deploy["spec"]["strategy"]["type"] == "Recreate"  # a single SQLite writer


def test_demo_train_registers_the_geo_l1_model(demo, monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "require", lambda *a: None)
    monkeypatch.setattr(demo.time, "sleep", lambda s: None)
    monkeypatch.setattr(demo.subprocess, "Popen", lambda cmd: type("P", (), {"terminate": lambda self: None})())
    monkeypatch.setattr(demo, "sh", lambda *cmd, **kw: calls.append((cmd, kw)))
    demo.train()
    (cmd, kw), = calls
    assert cmd[1:3] == ("-m", "training.train")
    assert "--geo" in cmd and cmd[cmd.index("--objective") + 1] == "l1"
    assert kw["env"]["MLFLOW_TRACKING_URI"] == "http://localhost:5000"
