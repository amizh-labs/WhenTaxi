"""Canary rollout demo driver: k3d + Argo Rollouts + Prometheus (cross-platform Python, pathlib only).

Run from WSL2 (Ubuntu) with Docker Desktop's WSL integration on, k3d and kubectl installed.
Every external command is printed exactly as it runs ("$ ..."), so you can follow along or re-run it.

    python k8s/demo.py up         # k3d cluster, Argo Rollouts, in-cluster MLflow and Prometheus
    python k8s/demo.py build      # build the API image and load it into the cluster
    python k8s/demo.py train      # train against the in-cluster MLflow (needs data/processed + data/feast)
    python k8s/demo.py deploy     # AnalysisTemplate, the Rollout (10/50/100 canary) and load traffic
    python k8s/demo.py good       # roll out a healthy new release  -> promoted to 100%
    python k8s/demo.py bad        # roll out a deliberately bad release (predictions x3) -> auto rollback
    python k8s/demo.py down       # delete the cluster

    --replicas 10 (default) or 5 (or env REPLICAS): API pod count. Without a service mesh or ingress,
    Argo splits traffic by pod count, so the 10/50/100 percentages are exact only with 10 replicas
    (1/5/10 pods). With 5 replicas they are approximate: the 10% step rounds up to 1 pod (~20%) and
    the 50% step becomes 2-3 pods. Each pod uses ~350 MB, so 5 replicas roughly halves memory.

    The first `deploy` never runs analysis: Argo promotes the very first revision straight to 100%
    because there is no stable version to compare with. `good` and `bad` each change the pod template
    (a new ETA_RELEASE value / ETA_FAULT=bias), which creates the new revision that actually gets
    canaried. If `good` shows no new revision, nothing was tested and the demo exits with an error.

    Suggested order (keep `kubectl argo rollouts get rollout eta-api -n eta --watch` open elsewhere):
    up, build, train, deploy --replicas 5, good (must reach 100%), bad (must abort, stable unchanged).
    If `good` aborts, run: kubectl describe analysisrun -n eta   (check the isNaN/len conditions first).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "k8s"
CLUSTER = "eta"
NAMESPACE = "eta"
ROLLOUT = "eta-api"
IMAGE = "eta-api:local"
ARGO_INSTALL = "https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml"
# The Argo CRDs exceed the 262144-byte annotation kubectl's client-side apply writes, so the install
# must be applied server-side. --force-conflicts lets a re-run take over fields from a failed first try.
ARGO_APPLY = ["kubectl", "apply", "--server-side", "--force-conflicts", "-n", "argo-rollouts", "-f", ARGO_INSTALL]
ALLOWED_REPLICAS = (5, 10)
DEFAULT_REPLICAS = 10


# ---- pure helpers (unit tested) -------------------------------------------------------------

def env_index(rollout: dict, name: str) -> int:
    env = rollout["spec"]["template"]["spec"]["containers"][0]["env"]
    return next(i for i, item in enumerate(env) if item["name"] == name)


def env_patch(rollout: dict, name: str, value: str) -> str:
    """JSON patch that sets one container env var, which changes the pod template -> new revision."""
    path = f"/spec/template/spec/containers/0/env/{env_index(rollout, name)}/value"
    return json.dumps([{"op": "replace", "path": path, "value": value}])


def env_patch_many(rollout: dict, changes: dict[str, str]) -> str:
    """JSON patch that sets several container env vars at once (one new revision)."""
    base = "/spec/template/spec/containers/0/env/"
    return json.dumps([{"op": "replace", "path": f"{base}{env_index(rollout, n)}/value", "value": v}
                       for n, v in changes.items()])


def classify(rollout: dict) -> str:
    """'aborted' (rolled back), 'healthy' (fully promoted) or 'progressing'."""
    status = rollout.get("status", {})
    if status.get("abort"):
        return "aborted"
    if status.get("phase") == "Healthy":
        return "healthy"
    return "progressing"


def revision_changed(before_hash: str | None, rollout: dict) -> bool:
    """True once Argo has noticed the patched pod template (a new pod-template hash became current).

    Right after `kubectl patch` the status can still say Healthy for the OLD revision; without this
    check the demo would report success before any rollout started, i.e. before anything was tested.
    """
    status = rollout.get("status", {})
    observed = status.get("observedGeneration") == str(rollout["metadata"]["generation"])
    return observed and status.get("currentPodHash") not in (None, before_hash)


def validate_replicas(replicas: int) -> int:
    if replicas not in ALLOWED_REPLICAS:
        raise ValueError(f"replicas must be one of {ALLOWED_REPLICAS}, got {replicas}")
    return replicas


def render_rollout(replicas: int, manifest: Path = K8S / "rollout.yaml") -> str:
    """rollout.yaml with spec.replicas set; every other document (the Service) is left untouched."""
    docs = list(yaml.safe_load_all(manifest.read_text()))
    for doc in docs:
        if doc and doc["kind"] == "Rollout":
            doc["spec"]["replicas"] = validate_replicas(replicas)
    return yaml.safe_dump_all([d for d in docs if d], sort_keys=False)


def cluster_config(repo: Path) -> str:
    """k3d config mounting the repo's weather file and Feast store into every node (served via hostPath)."""
    return f"""apiVersion: k3d.io/v1alpha5
kind: Simple
metadata:
  name: {CLUSTER}
servers: 1
agents: 2
volumes:
  - volume: {(repo / "data" / "raw").as_posix()}:/data/raw
    nodeFilters: [all]
  - volume: {(repo / "data" / "feast").as_posix()}:/data/feast
    nodeFilters: [all]
"""


# ---- commands -------------------------------------------------------------------------------

def sh(*cmd: str, check: bool = True, capture: bool = False, env: dict | None = None, quiet: bool = False):
    """Run a command, printing it first exactly as executed (unless quiet, for repeated polling)."""
    if not quiet:
        print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True, capture_output=capture, env=env)


def write_temp(content: str, suffix: str, label: str) -> str:
    """Write a generated file and print where it is, so you can open it before it is used."""
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as handle:
        handle.write(content)
    print(f"# generated {label}: {handle.name}", flush=True)
    return handle.name


def require(*tools: str) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        sys.exit(f"missing tools: {', '.join(missing)} (run from WSL2 with Docker Desktop, k3d and kubectl)")


def rollout(quiet: bool = False) -> dict:
    out = sh("kubectl", "get", "rollout", ROLLOUT, "-n", NAMESPACE, "-o", "json", capture=True, quiet=quiet)
    return json.loads(out.stdout)


def up() -> None:
    require("docker", "k3d", "kubectl")
    for sub in ("raw", "feast"):
        (ROOT / "data" / sub).mkdir(parents=True, exist_ok=True)
    # Re-runnable: only create the cluster if it does not exist yet.
    if sh("k3d", "cluster", "get", CLUSTER, check=False, capture=True).returncode == 0:
        print(f"# k3d cluster '{CLUSTER}' already exists, reusing it", flush=True)
    else:
        sh("k3d", "cluster", "create", "--config", write_temp(cluster_config(ROOT), ".yaml", "k3d config"))
    sh("kubectl", "create", "namespace", "argo-rollouts", check=False)
    sh(*ARGO_APPLY)
    sh("kubectl", "create", "namespace", NAMESPACE, check=False)
    sh("kubectl", "apply", "-f", str(K8S / "mlflow.yaml"), "-f", str(K8S / "prometheus.yaml"))
    sh("kubectl", "rollout", "status", "deployment/mlflow", "-n", NAMESPACE, "--timeout=300s", check=False)
    sh("kubectl", "rollout", "status", "deployment/prometheus", "-n", NAMESPACE, "--timeout=300s", check=False)
    sh("kubectl", "wait", "--for=condition=available", "deployment/argo-rollouts", "-n", "argo-rollouts",
       "--timeout=300s")


def build() -> None:
    require("docker", "k3d")
    sh("docker", "build", "-t", IMAGE, str(ROOT))
    sh("k3d", "image", "import", IMAGE, "-c", CLUSTER)


# The shipped configuration (see docs/contracts.md section 6): geo features + L1 objective.
TRAIN_ARGS = ["--geo", "--objective", "l1"]


def train() -> None:
    """Train against the in-cluster MLflow through a temporary port-forward."""
    require("kubectl")
    cmd = ["kubectl", "port-forward", "svc/mlflow", "5000:5000", "-n", NAMESPACE]
    print("$", " ".join(cmd), "  # background, stopped after training", flush=True)
    forward = subprocess.Popen(cmd)
    try:
        time.sleep(5)
        print("# MLFLOW_TRACKING_URI=http://localhost:5000", flush=True)
        sh(sys.executable, "-m", "training.train", *TRAIN_ARGS, env={**os.environ, "MLFLOW_TRACKING_URI": "http://localhost:5000"})
    finally:
        forward.terminate()


def deploy(replicas: int) -> None:
    require("kubectl")
    rendered = write_temp(render_rollout(replicas), ".yaml", f"rollout (replicas={replicas})")
    sh("kubectl", "apply", "-f", str(K8S / "analysis-template.yaml"), "-f", rendered, "-f", str(K8S / "traffic.yaml"))
    wait_for(timeout=600)


def wait_for(timeout: int) -> str:
    deadline, state, first = time.time() + timeout, "progressing", True
    while time.time() < deadline:
        state = classify(rollout(quiet=not first))
        if first:
            print("# polling the same command every 10s", flush=True)
            first = False
        print(f"  rollout state: {state}", flush=True)
        if state in ("healthy", "aborted"):
            break
        time.sleep(10)
    return state


def release(kind: str) -> None:
    require("kubectl")
    current = rollout()
    before = current["status"].get("stableRS")
    before_current = current["status"].get("currentPodHash")
    # A good release must also clear ETA_FAULT: an aborted bad rollout leaves it set in the pod template.
    changes = {"ETA_FAULT": "bias"} if kind == "bad" else {"ETA_RELEASE": str(int(time.time())), "ETA_FAULT": ""}
    sh("kubectl", "patch", "rollout", ROLLOUT, "-n", NAMESPACE, "--type=json", "-p", env_patch_many(current, changes))

    # The first deploy goes straight to 100% (no stable version to compare with); only a NEW revision
    # starts a canary. Wait until Argo has actually picked up the changed pod template.
    print("# waiting for Argo to register the new revision (pod template changed)", flush=True)
    deadline = time.time() + 90
    while not revision_changed(before_current, rollout(quiet=True)):
        if time.time() > deadline:
            sys.exit("NO NEW REVISION: the rollout did not change, so nothing was tested")
        time.sleep(3)
    state = wait_for(timeout=900)
    after = rollout(quiet=True)["status"].get("stableRS")
    print(f"\nresult: {state}; stable ReplicaSet before={before} after={after}")
    sh("kubectl", "get", "analysisrun", "-n", NAMESPACE, check=False)
    if kind == "bad":
        ok = state == "aborted" and before == after
        print("ROLLBACK OK: bad release aborted, stable version unchanged" if ok else "ROLLBACK FAILED")
        sys.exit(0 if ok else 1)
    ok = state == "healthy" and before != after
    print("PROMOTED: good release is now stable" if ok else "PROMOTION FAILED")
    sys.exit(0 if ok else 1)


def down() -> None:
    require("k3d")
    sh("k3d", "cluster", "delete", CLUSTER)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["up", "build", "train", "deploy", "good", "bad", "down"])
    parser.add_argument("--replicas", type=int, choices=ALLOWED_REPLICAS,
                        default=int(os.environ.get("REPLICAS", DEFAULT_REPLICAS)))
    args = parser.parse_args()
    {"up": up, "build": build, "train": train, "deploy": lambda: deploy(args.replicas), "down": down,
     "good": lambda: release("good"), "bad": lambda: release("bad")}[args.command]()
