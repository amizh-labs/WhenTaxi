# Runbook: fresh clone to a working `/predict`

Windows + PowerShell, Python 3.11 or 3.12 (the project needs `>=3.11,<3.13`; check with `py --list`).
Every command is run from the repository root. Nothing here needs Docker.

## Quickstart

```powershell
git clone https://github.com/amizh-labs/ETA.git
cd ETA

# 1. environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[serving,flows,monitoring,dev]"

# 2. raw data -> data/raw (3 trip files ~3.7-4M rows each, zone lookup, weather)
python scripts/download_data.py

# 3. clean + validate -> data/processed/trips_clean.parquet and data_quality_report.json
python -m validation.clean

# 4. Feast online store (rolling zone-pair features) -> data/feast
python -m features.store.materialize

# 5. train on Jan-Feb, validate on Mar; registers model `eta-duration` and sets alias `champion` -> mlflow.db
python -m training.train

# 6. serve
uvicorn serving.app:create_app --factory --port 8000
```

In a second PowerShell window:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" `
  -Body '{"pickup_zone": 161, "dropoff_zone": 237, "pickup_ts": "2026-03-20T08:30:00"}'
```

Expected: `/health` returns `status: ok` with the model version, and `/predict` returns `eta_minutes` (a number of
minutes, about 8 for this pair) plus `model_version`.

Steps 3-5 can also run as one Prefect flow: `python -m flows.training_flow`.
For a quicker, slightly weaker model add `--max-train-rows 500000` to step 5.

## Check it

```powershell
pytest                       # unit and integration tests
ruff check .
python scripts/verify.py     # leakage, time split, weather NaN, baseline, seed, unseen-zone (trains twice; a few minutes)
```

## Files a fresh clone needs that are NOT in git

| Path | Why it is not in git | How to create it |
|---|---|---|
| `.venv/` | local virtualenv | step 1 |
| `data/raw/yellow_tripdata_2026-01.parquet`, `-02`, `-03` | large public data, DVC-tracked, never in git | `python scripts/download_data.py` |
| `data/raw/taxi_zone_lookup.csv` | same | `python scripts/download_data.py` |
| `data/raw/weather_2026_01_07.json` | same (Open-Meteo, NYC local time) | `python scripts/download_data.py` |
| `data/processed/trips_clean.parquet`, `data/processed/data_quality_report.json` | generated (~94 MB) | `python -m validation.clean` |
| `data/feast/registry.db`, `online.db`, `snapshots.parquet` | generated Feast store (~7 MB, binary) | `python -m features.store.materialize` |
| `mlflow.db` (plus `mlruns/`, `mlartifacts/` if used) | local MLflow tracking store and model registry | `python -m training.train` |
| `src/eta_mlops.egg-info/` | created by `pip install -e` | step 1 |

Without `data/feast` the API still starts (the rolling features become NaN and the model degrades gracefully), and
without `mlflow.db`/the `champion` alias it refuses to start. The weather file is read at startup from
`data/raw/` (override with `ETA_WEATHER_PATH`).

Not needed for this quickstart: `.env` (ignored, unused), Docker images, the k3d cluster.
`data/raw` is meant to be DVC-tracked, but `dvc init` has not been run yet, so `scripts/download_data.py` is the only source.

## Optional: the canary demo (WSL2 + Docker Desktop)

See the header of `k8s/demo.py`: `up`, `build`, `train`, `deploy --replicas 5`, `good`, `bad`.
`k8s/mlflow.yaml` stores the registry on a PersistentVolumeClaim, so it survives pod restarts but not
`k3d cluster delete`; re-run `python k8s/demo.py train` after recreating the cluster.

## The evening-rush surge demo (WSL2 + Docker Desktop)

Shows the platform **detecting a traffic shift and retraining a challenger**. It does not show the model adapting. The
surge window's durations are **SYNTHETIC** (true durations x1.4). Needs the held-out April file at
`data/holdout/yellow_tripdata_2026-04.parquet` (outside `data/raw`, so DVC and the cleaning stage are untouched), the cleaned
Jan-Mar data (`data/processed/trips_clean.parquet`) and the raw files in `data/raw`. Run in WSL, from the repo root, venv active:

```bash
ls data/holdout/yellow_tripdata_2026-04.parquet data/processed/trips_clean.parquet   # inputs exist
docker compose up -d                                                                  # API (prediction log on), MLflow, Prometheus, Grafana
MLFLOW_TRACKING_URI=http://localhost:5000 python -m training.train --geo --objective l1   # first time only: a champion in the compose MLflow
MLFLOW_TRACKING_URI=http://localhost:5000 python -m monitoring.surge_demo             # the demo (fast retrain on a row sample)
docker compose down
```

Options: `--n 2000` requests per window, `--retrain-rows 300000` (training-row sample) or `--full-retrain`, `--no-retrain`,
`--mae-threshold 0.10`, `--seed 0`, `--url http://localhost:8000`, `--holdout PATH`. The demo stops and restarts the `api`
container around each window to read its prediction log safely. It prints the summary table and writes
`data/processed/surge_demo_report.json`; per-window drift reports are in `data/processed/surge_demo/`.

What it never does: write to the real MLflow registry or the `champion` alias (the champion is only read; the retrain uses a
throwaway SQLite store under `data/processed/surge_demo/retrain/`), or touch `data/raw`, `data/processed/trips_clean.parquet`
or `data/feast`. The throwaway store has no champion, so the retrain flow's new-month rule fires as well as drift; the report
says so.

