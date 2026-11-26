# Part A Handoff (for amizh)

You own Part A. vaish owns Part B and is building it in parallel. Both of us work in the
common repo `https://github.com/amizh-labs/ETA`. Read `CLAUDE.md` and `docs/contracts.md` first.

## 1. Project context

An end-to-end MLOps platform that predicts NYC yellow taxi trip duration (ETA).
**The platform is the point** (CI/CD, canary, drift, retraining). The model stays simple (LightGBM).

### Constraints
- Zero-cost, open-source only. No AWS/GCP/Azure, no paid services.
- Windows, 24 GB RAM, Docker Desktop. Use `pathlib`; no bash-only scripts.
- Stack (do not add tools outside it): DVC, Pandera, Feast, Prefect, MLflow, LightGBM, FastAPI,
  Docker, GitHub Actions, k3d + Argo Rollouts, Prometheus, Grafana, Evidently.
- Every stage gets tests. After each stage, write a short explanation of what you built.

### Data (`data/raw`, tracked by DVC, never committed to git)
- `yellow_tripdata_2026-01/02/03.parquet` (~3.7-4M rows each). Apr-Jul 2026 are held out as
  "new monthly data" for later retraining/drift demos.
- `taxi_zone_lookup.csv` (LocationID, Borough, Zone, service_zone)
- `weather_2026_01_07.json` (Open-Meteo hourly: temperature_2m, precipitation, snowfall,
  wind_speed_10m; NYC local time)
- Get the files with `python scripts/download_data.py`.
- Columns used: `tpep_pickup_datetime`, `tpep_dropoff_datetime`, `PULocationID`, `DOLocationID`.
- Target: `duration_min = dropoff - pickup`.
- Known problems: pickups outside the file's month (even year 2008), zero/negative/huge
  durations, zone IDs not in the lookup. Cleaning must drop these and write a per-rule report.
- 2025+ files have an extra `cbd_congestion_fee` column. Never mix 2024 and 2026 data.

### Modeling rules
- Split by time: train Jan-Feb, validate Mar. Never random.
- Do NOT use `trip_distance`, `fare_amount`, `total_amount` (known only after the trip = leakage).
- Request-time features only: pickup zone, dropoff zone, hour, day of week, weekend, weather.
- Metrics: MAE (headline), RMSE, p90 absolute error. Always compare against a zone-pair-mean baseline.
- Feature code lives in ONE module (`src/features/`, owned by vaish) that training and serving
  both import.

## 2. Who owns what

| Area | You (A) | vaish (B) |
|---|---|---|
| Data | Ingestion and quality | Features, Feast, skew test |
| ML | Training and monitoring | Evaluation and promotion |
| MLOps | Orchestration (Prefect) | CI/CD (GitHub Actions) |
| Deployment | Serving and observability | Safe rollout (k3d, Argo) |

Your directories: `src/validation/`, `src/training/`, `flows/`, `src/serving/`, `monitoring/`.
Do not edit vaish's directories (`src/features/`, `src/evaluation/`, `.github/`, `k8s/`);
ask in a PR comment or issue instead.

## 3. Working rules
- Run Claude Code (or any tool) on YOUR machine with YOUR git identity. Commit only as yourself.
- Branch per task, e.g. `a/validation`. Open a PR to `main`; vaish reviews, and you review theirs.
  The reviewer must be able to explain the change.
- Never commit anything in `data/raw`. Track it with DVC.
- Changes to `docs/contracts.md` need both of us to approve.
- Use `Co-authored-by:` trailers for work we do together (integration points).

## 4. Your tasks, in order

### A1. Ingestion and quality (`src/validation/`) - start here
1. `dvc init`; track `data/raw`. Use a free local DVC remote (a folder outside the repo) for now.
2. Cleaning module: load a month, compute `duration_min`, apply rules, and drop:
   - pickup outside the file's month
   - duration <= 0 or above a configured maximum (e.g. 180 min; keep it in config)
   - PU/DO zone IDs not in `taxi_zone_lookup.csv`
3. Output the cleaned table matching the contract in `docs/contracts.md` section 1:
   `pickup_ts, PULocationID, DOLocationID, duration_min, month`, written as parquet to `data/processed/`.
4. Pandera schema validating the cleaned table.
5. Data-quality report: rows in, rows dropped per rule, rows out, as JSON (and a readable summary).
6. Tests: tiny synthetic frames covering each rule, the schema, and the report counts.

**Done when:** `python -m` entry point cleans Jan, Feb and Mar, report numbers add up, tests pass.
vaish builds features against the cleaned-table contract from day one, so keep the schema stable.

### A2. Training and tracking (`src/training/`)
1. Zone-pair-mean baseline (mean duration per (PU, DO) pair, with a global-mean fallback for unseen pairs).
2. LightGBM using features from `src/features/` (vaish). Until it lands, use a temporary local
   feature function behind the same function signature, and swap when vaish's module merges.
3. MLflow: log params, MAE/RMSE/p90 for baseline and LightGBM, and the model artifact.
4. Register the model per `docs/contracts.md` section 2 (name, `champion` alias).
5. Tests: baseline fallback, metric functions, a tiny-data training smoke test.

### A3. Serving and observability (`src/serving/`, `monitoring/`)
1. FastAPI `POST /predict` per contract section 3; load the registered model at startup; also `/health`.
2. Prometheus metrics with the contract names (section 4) via `prometheus-client`, at `/metrics`.
3. Dockerfile for the API, and `docker-compose.yml` with API, MLflow, Prometheus and Grafana.
4. Grafana dashboards: request rate, latency, prediction distribution, errors.
5. Traffic simulator script that replays held-out data against `/predict` (the evening-rush shift scenario is
   `monitoring/surge_demo.py`).
6. Tests: endpoint with a stub model, metrics are exposed, input validation.

### A4. Drift and orchestration (`flows/`, `monitoring/`)
1. Evidently drift report comparing a reference month with a new month; write the JSON output
   in the contract format (section 5).
2. Prefect training flow: ingest -> clean/validate -> features -> train -> log/register.
3. Retrain trigger: new month of data arrives OR drift flag is true.
4. Tests: flow runs end to end on a small sample; drift output matches the contract.

### What vaish provides you
- `src/features/`: the shared feature code (early, on a small sample).
- Feast stores and skew test, evaluation/promotion logic, CI (lint, tests, metric gate), k8s/Argo canary.
- Argo's AnalysisTemplate queries YOUR Prometheus metric names, so do not rename them without
  a contracts PR.

## 5. Suggested order
A1 -> A2 -> A3 -> A4. Open a PR after each task, not one big PR at the end.
