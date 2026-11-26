# WhenTaxi

**An end-to-end MLOps platform for taxi trip ETA prediction.**
Versioned data, a feature store, gated CI, canary rollouts that roll back on their own, and drift monitoring, built around a deliberately simple LightGBM model.

[![CI](https://github.com/amizh-labs/WhenTaxi/actions/workflows/ci.yml/badge.svg)](https://github.com/amizh-labs/WhenTaxi/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Stack](https://img.shields.io/badge/stack-DVC%20%7C%20Feast%20%7C%20MLflow%20%7C%20FastAPI%20%7C%20Argo%20Rollouts-informational)

---

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech stack](#tech-stack)
4. [Results and metrics](#results-and-metrics)
5. [ML design](#ml-design)
6. [MLOps lifecycle](#mlops-lifecycle)
7. [Monitoring and alerting](#monitoring-and-alerting)
8. [API reference](#api-reference)
9. [Getting started](#getting-started)
10. [Repository structure](#repository-structure)
11. [Testing and quality](#testing-and-quality)
12. [Team and ownership](#team-and-ownership)
13. [Data sources](#data-sources)

---

## Overview

Given a pickup zone, a dropoff zone, and a pickup time, WhenTaxi predicts the trip duration in minutes. The model is a LightGBM regressor. The point of the project is everything around it:

| Concern | What the platform does |
|---|---|
| Data quality | Cleans raw TLC files with per-rule drop reporting and Pandera schema validation |
| Reproducibility | DVC versions the raw data; `dvc repro` rebuilds the cleaned table |
| Training and serving consistency | One shared feature module, a Feast feature store, and an automated offline vs online skew test |
| Experiment tracking | MLflow runs, metrics, artifacts, and a model registry with `champion` and `challenger` aliases |
| Change safety | A CI metric-regression gate and an Argo Rollouts canary that aborts automatically on bad releases |
| Observability | Prometheus metrics, Grafana dashboards, drift reports on real requests, and feature-freshness alerts |
| Automation | Prefect flows for retraining and scheduled drift checks, plus a monthly data refresh flow |

### Key features

- **Leakage-free modeling.** Post-trip fields (`trip_distance`, fare, total, tip) are never used. Evaluation is a strict time-based split.
- **Training-serving skew test.** CI fails if offline and online feature values differ for the same entity and timestamp.
- **Fail-closed canary analysis.** Empty or NaN metrics can never count as a pass.
- **Fail-open request logging.** The prediction log cannot slow down or break `/predict`.
- **Size-aware drift detection.** Per-column KS tests with an effect-size floor, calibrated against a measured false-alarm rate.
- **Explicit data-quality signals.** A missing weather feed is reported as a data-quality alert and never as drift.

---

## Architecture

```mermaid
flowchart TB
  subgraph DATA["Data layer"]
    TLC["NYC TLC monthly trip files"]
    WX["Open-Meteo weather"]
    DVC["DVC versioned raw data"]
    CLEAN["Cleaning and Pandera validation"]
    TRIPS[("trips_clean.parquet")]
    TLC --> DVC
    WX --> DVC
    DVC --> CLEAN --> TRIPS
  end

  subgraph FEAT["Feature layer"]
    BUILD["Shared feature code: time, weather, geo"]
    FEAST[("Feast offline and online stores")]
    SKEW["Offline vs online skew test"]
  end

  subgraph ML["Training and registry"]
    TRAIN["LightGBM training, time-based split"]
    MLFLOW[("MLflow tracking and model registry")]
    GATE["Champion vs challenger evaluation"]
  end

  subgraph SERVE["Serving"]
    API["FastAPI predict service"]
    LOG[("Prediction log, SQLite")]
    UI["Streamlit demo UI"]
  end

  subgraph DELIVERY["Delivery"]
    CI["GitHub Actions: ruff, pytest, metric gate, image build"]
    GHCR["Container registry"]
    ARGO["Argo Rollouts canary on k3d"]
  end

  subgraph OBS["Monitoring"]
    PROM["Prometheus"]
    GRAF["Grafana dashboards"]
    DRIFT["Drift report: Evidently with KS tests"]
    ALERTS["Alert rules: feature freshness"]
  end

  subgraph ORCH["Orchestration"]
    FLOWS["Prefect flows: retrain, drift, monthly refresh"]
  end

  TRIPS --> BUILD --> FEAST
  FEAST --> SKEW
  TRIPS --> TRAIN
  BUILD --> TRAIN
  TRAIN --> MLFLOW --> GATE
  GATE --> CI --> GHCR --> ARGO
  MLFLOW --> API
  FEAST --> API
  UI --> API
  API --> LOG --> DRIFT
  API --> PROM --> GRAF
  PROM --> ARGO
  PROM --> ALERTS
  DRIFT --> FLOWS
  FLOWS --> TRAIN
```

### Release flow

```mermaid
flowchart LR
  A["New model registered as challenger"] --> B["CI: metric-regression gate"]
  B -->|pass| C["Promotion: champion alias"]
  B -->|fail| X["PR blocked"]
  C --> D["Canary at 10 percent"]
  D --> E{"Prometheus analysis"}
  E -->|"all checks pass"| F["50 percent, then 100 percent"]
  E -->|"any threshold breach"| G["Abort: stable keeps serving"]
  E -->|"no data three times"| H["Pause for a human decision"]
```

### Components

| Component | Responsibility | Location |
|---|---|---|
| Cleaning and validation | Drop out-of-month, impossible-duration, and unknown-zone trips; write a per-rule report | `src/validation/` |
| Feature module | One function for time, weather, and geo features, shared by training and serving | `src/features/` |
| Feature store | Rolling zone-pair features, offline and online stores, freshness metadata | `src/features/store/` |
| Training | Zone-pair baseline, LightGBM, time-based split, MLflow logging and registration | `src/training/` |
| Evaluation | Champion vs challenger comparison, error analysis, CI metric gate, promotion | `src/evaluation/` |
| Serving | FastAPI `/predict`, live weather with fallback, Prometheus metrics, prediction log | `src/serving/` |
| Monitoring | Drift on logged requests, traffic simulator, surge demo, alert rules, Grafana | `monitoring/` |
| Flows | Retrain, scheduled drift check, processed-months state, monthly refresh | `flows/` |
| Cluster | k3d manifests, Argo Rollout, analysis template, demo driver | `k8s/` |
| Demo UI | Streamlit page that calls the API | `ui/` |

---

## Tech stack

| Layer | Tools | Role |
|---|---|---|
| Language and packaging | Python 3.11, `pyproject.toml` with version bounds (for example `pandas<3`) | Reproducible installs |
| Data versioning | DVC | Versioned raw data and the `clean` pipeline stage |
| Data validation | Pandera, pandas, PyArrow | Schema checks and cleaning |
| Feature store | Feast (SQLite offline and online stores) | Rolling zone-pair features, point-in-time correctness |
| Modeling | LightGBM, scikit-learn | Regression model and metrics |
| Experiment tracking and registry | MLflow 3.x (SQLite backend) | Runs, artifacts, `champion` and `challenger` aliases |
| Serving | FastAPI, Uvicorn, Docker | Prediction API and container image |
| Orchestration | Prefect | Retrain, drift, and refresh flows |
| CI/CD | GitHub Actions, GitHub Container Registry | Lint, tests, metric gate, image build |
| Deployment | k3d (Kubernetes in Docker), Argo Rollouts | Canary rollout with automatic rollback |
| Metrics and dashboards | Prometheus, Grafana | Latency, errors, prediction and feature-age metrics |
| Drift detection | Evidently, SciPy (KS tests) | Drift reports on logged requests |
| Demo UI | Streamlit (separate image) | Try the API in a browser |
| Quality | pytest, Ruff | 290+ tests (plus 16 UI tests) and linting |
| External data | NYC TLC trip records, TLC zone shapefile, Open-Meteo | Trips, zone geometry, weather |

All tools are free and open source. The platform runs on a single laptop (24 GB RAM recommended for the full cluster demo).

---

## Results and metrics

All numbers below were measured on real data. The validation month is **March 2026**, a month the models never trained on. Training data is January and February 2026.

### Data volumes (after cleaning)

| Split | Months | Cleaned trips |
|---|---|---|
| Train | Jan-Feb 2026 | 6.92 M |
| Validation | Mar 2026 | 3.84 M |
| Holdout (surge and refresh demos) | Apr 2026 | about 3.7 M |

### Model performance on March 2026

| Model | MAE (min) | RMSE (min) | p90 abs error (min) |
|---|---|---|---|
| Always predict the median (13.5 min) | 9.10 | n/a | n/a |
| Zone-pair mean baseline | 4.75 | 7.63 | 9.67 |
| LightGBM, 12 features, squared error | 3.88 | 6.46 | 7.95 |
| LightGBM, geo features and L1 objective (the shipped champion) | **3.79** | 6.55 | 7.82 |

The champion is about 20% better than the baseline on MAE. L1 optimizes MAE directly, which is why its RMSE is slightly above the squared-error model's.

### Ablation (two time-based folds, 2M-row training samples)

Fold A trains on January and validates on February. Fold B trains on January-February and validates on March.

| Variant | Fold A MAE | Fold B MAE | Mean MAE |
|---|---|---|---|
| 12 features, squared error | 4.1305 | 3.9407 | 4.0356 |
| plus geo features | 4.0553 | 3.8589 | 3.9571 |
| plus geo features and **L1** | 4.0310 | 3.8127 | **3.9218** |
| plus geo features and Huber (delta 5 min) | 4.0345 | 3.8108 | 3.9226 |
| plus geo features and log1p target | 4.0516 | 3.8248 | 3.9382 |

Geo features give about 2.0% lower MAE, and the L1 objective adds about 0.9%. L1 and Huber are a statistical tie.

### Error by slice (fold B, champion configuration)

| Slice | Trips | MAE (min) |
|---|---|---|
| Manhattan | 3.31 M | 3.35 |
| Queens | 330 k | 6.12 |
| Brooklyn | 158 k | 7.10 |
| Bronx | 39 k | 9.92 |
| Non-airport trips | 3.55 M | 3.64 |
| Airport trips | 294 k | 5.92 |

Airport trips are the one slice where the geo and L1 changes did not help (5.87 before, 5.92 after). The Staten Island and EWR slices are too small to interpret.

### Feature importance (by gain)

| Feature group | Share |
|---|---|
| `avg_duration_24h` | about 66% |
| Zone-to-zone distance | about 10% |
| Hour of day | about 6% |
| `avg_duration_1h` | about 6% |
| Weather (all four columns) | 1.7% |

The model is largely **weather-insensitive**: weather features account for 1.7% of importance, and the training months contain almost no extreme weather to learn from.

### Month-over-month check

The champion scored **4.08 MAE on the full April 2026 holdout**, 7.7% worse than March. This is the reason the platform re-evaluates on every newly published month.

### Canary analysis thresholds

| Check | Condition for a pass |
|---|---|
| p95 request latency (canary pods only) | below 0.5 s |
| 5xx error rate | below 2% |
| Mean predicted ETA vs the stable version | within 25% |
| Missing or NaN metric | never a pass |

In a fault-injection test on an earlier manifest revision, a release with predictions multiplied by 3 read a canary/stable ratio of 2.08, 2.25, and 2.32 against a limit of 0.25 and was aborted while latency and error rate stayed clean. The good and bad runs were then repeated on the final manifests (sanity comparison over a 3-minute window): a healthy release promoted to 100%, and a faulty one was aborted and rolled back.

### Drift detector calibration

False-alarm rate measured by drawing 50 random windows from the reference data itself:

| Window size | `dataset_drift` fired |
|---|---|
| 200 rows | 2% (1 of 50) |
| 500 rows | 0% (0 of 50) |
| 2000 rows | 0% (0 of 50) |

Evidently's default distance threshold fired in 26 of 50 windows at 200 rows, so per-column KS tests (p < 0.003) with an effect-size floor are used instead.

### Evening-rush surge demo (2,000 replayed trips per window)

| Window | MAE (min) | Dataset drift | Drifted trigger columns |
|---|---|---|---|
| Baseline (April as is) | 4.03 | false | none |
| Surge (evening and airport heavy) | 10.91 (against synthetic, inflated durations) | true | `hour`, `distance_km` |

The surge MAE is built in by construction (durations inflated by 40%) and is not a performance figure. The demo shows detection and retraining, not model adaptation.

---

## ML design

**Target.** `duration_min = dropoff time - pickup time`, kept between 1 and 180 minutes.

**Cleaning rules** (each reported separately in `data_quality_report.json`):

1. Pickup must fall inside the file's own month (raw files contain out-of-month pickups, including timestamps from 2008).
2. Dropoff must be after pickup and the duration within 1 to 180 minutes.
3. Both zone IDs must exist in `taxi_zone_lookup.csv`.
4. Rows pointing to "Unknown" or "Outside of NYC" zones are dropped under their own rule.

**Features available at request time only**

| Group | Features |
|---|---|
| Location | Pickup zone and dropoff zone (categorical, fixed category list from the lookup table) |
| Time | Hour, day of week, weekend flag |
| Weather | Temperature, precipitation, snowfall, wind speed (Open-Meteo hourly) |
| Rolling traffic (Feast) | Average duration over the last 1 hour and 24 hours, trip count over 24 hours, per zone pair, computed point-in-time |
| Geo | Haversine distance and bearing between zone centroids, same-zone flag, borough pair, airport flag |

**Modeling rules**

- Time-based split only, never random.
- No post-trip fields: `trip_distance`, `fare_amount`, `total_amount`, and `tip_amount` are banned and checked by `scripts/verify.py`.
- Always compare against the zone-pair-mean baseline.
- Feature code lives in one module imported by both training and serving.
- Zone category lists are fixed from the lookup table, so an unseen zone at serving time never silently shifts category codes.

**Model.** LightGBM regressor with the L1 objective, tuned to the headline metric (MAE).

---

## MLOps lifecycle

### Data and features

- `data/raw` is tracked by DVC (`data/raw.dvc`). The `clean` stage in `dvc.yaml` builds `data/processed/trips_clean.parquet`, and a second `dvc repro` is a no-op until inputs change.
- Feast materializes rolling zone-pair features. `materialized.json` records when, and the API exports the store's age as a metric.
- The skew test compares offline and online feature values for the same entity and timestamp.

### Experiment tracking and registry

- Every training run logs parameters, baseline and model metrics, feature importance, the data-quality report, and the train and validation months to MLflow.
- Registered model name: `eta-duration`. The newest version gets the `challenger` alias. The `champion` alias is only ever moved by promotion logic.

### CI

On every push and pull request, GitHub Actions:

1. Installs the project from a clean environment.
2. Runs Ruff and the full pytest suite.
3. Trains the shipped configuration on a fixed sample and **fails if MAE regresses** against a stored reference file.
4. Builds the serving image (and publishes it on merge to `main`).
5. Runs the UI tests in a separate job that installs only `ui/requirements.txt`, and fails on any skipped test.

### Progressive delivery

Argo Rollouts shifts traffic 10%, then 50%, then 100% with timed pauses. A background analysis queries Prometheus for the canary pods only. Real threshold breaches fail on the first measurement. Empty or NaN results are *inconclusive*: after three, the rollout pauses for a human, so a canary with no data is never promoted and never silently passes.

### Automation

| Flow | Purpose |
|---|---|
| `flows/retrain.py`, `flows/training_flow.py` | Train and register a challenger; retrain when drift or an accuracy flag fires |
| `flows/drift_flow.py` | Daily drift check on the prediction log (`--serve` schedules it at 06:00) |
| `flows/month_state.py` | Atomic state file of processed months, updated only after a month fully succeeds |
| `flows/refresh_flow.py` | Monthly refresh: detect a newly published month, clean, score the champion, retrain (written and mock-tested; not yet run on real data) |

---

## Monitoring and alerting

### Prometheus metrics (exported by the API)

| Metric | Type | Meaning |
|---|---|---|
| `eta_request_latency_seconds` | histogram | End-to-end `/predict` latency |
| `eta_requests_total` | counter (label `status`) | Requests by HTTP status |
| `eta_prediction_minutes` | histogram | Distribution of predicted ETAs |
| `eta_feature_age_seconds` | gauge | Seconds since the Feast online store was materialized (-1 if unknown) |
| `eta_feature_age_alert_threshold_seconds` | gauge | Configured staleness threshold |

### Alert rules

| Alert | Fires when |
|---|---|
| `EtaFeaturesStale` | Feature age exceeds the threshold (default 7 days) |
| `EtaFeaturesUnknown` | The feature store was never materialized or its metadata is unreadable |

### Grafana dashboard

Request rate, p95 latency, 5xx ratio, share of predictions over 30 minutes, feature age vs threshold, firing alerts, and a predicted-ETA distribution heatmap. Drift is not on the dashboard: the API exports no drift metrics, so drift lives in the drift reports.

### Drift detection

`monitoring/prediction_drift.py` compares the logged request window against a 20,000-row Jan-Feb reference.

- **Trigger columns** (can set `dataset_drift`): `hour`, day of week, and `distance_km`.
- **Context columns** (reported, never trigger): the four weather columns.
- **Data quality** (separate alert): share of requests with missing weather, flagged above 20%.
- Fewer than 200 logged rows returns `insufficient data` and never flags drift.

---

## API reference

### `POST /predict`

```json
{
  "pickup_zone": 161,
  "dropoff_zone": 237,
  "pickup_ts": "2026-03-10T08:30:00"
}
```

```json
{
  "eta_minutes": 8.4,
  "model_version": "1",
  "weather_source": "archive"
}
```

Optional fields let a caller supply the four weather values directly.

| `weather_source` | Meaning |
|---|---|
| `request` | Supplied by the caller |
| `live` | Current Open-Meteo forecast for a pickup time from yesterday to two days ahead |
| `archive` | Stored historical weather file |
| `missing` | No weather available; the model receives missing values |

Weather resolution order: request, live, archive, missing. A 2-second timeout and a short failure memory mean a weather outage cannot stall requests.

### `GET /metrics`

Prometheus exposition format (metrics above).

### `GET /docs`

Interactive OpenAPI documentation (FastAPI).

---

## Getting started

### Prerequisites

- Python 3.11
- Docker Desktop with the WSL 2 backend (Windows) or Docker Engine (Linux/macOS)
- For the cluster demo: `kubectl`, `k3d`, and the `kubectl-argo-rollouts` plugin, run inside WSL 2 or Linux

### 1. Install

```bash
git clone https://github.com/amizh-labs/WhenTaxi.git
cd WhenTaxi
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serving,flows,monitoring]"
```

### 2. Get the data

```bash
python scripts/download_data.py        # TLC trips, zone lookup, weather
dvc add data/raw                       # track it (identical files give an identical hash)
dvc repro                              # clean stage: writes data/processed/trips_clean.parquet
```

### 3. Verify the pipeline

```bash
pytest -q && ruff check .
python scripts/verify.py               # leakage, time split, weather NaN rate, baseline, seed, unseen zone
```

### 4. Train and serve

```bash
python -m features.store.materialize   # build the Feast stores
python -m training.train --geo --objective l1
docker compose up --build              # MLflow, API, Prometheus, Grafana, UI
```

When training against the Compose MLflow, set `MLFLOW_TRACKING_URI=http://localhost:5000` first.

| Service | URL |
|---|---|
| API and docs | http://localhost:8000 and http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| Demo UI | http://localhost:8501 |

```bash
curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"pickup_zone":161,"dropoff_zone":237,"pickup_ts":"2026-03-10T08:30:00"}'
```

The exact end-to-end command sequence, with expected output for each step, is in `docs/RUNBOOK.md`.

### 5. Canary demo on Kubernetes (WSL 2)

```bash
python k8s/demo.py up
python k8s/demo.py build
python k8s/demo.py train
python k8s/demo.py deploy --replicas 5
python k8s/demo.py good        # a healthy release promotes to 100%
python k8s/demo.py bad         # predictions inflated 3x: analysis aborts, stable keeps serving
python k8s/demo.py down
```

`--replicas 5` halves memory use. Without a service mesh, traffic splits by pod count, so at 5 replicas the canary steps are approximately (not exactly) 10, 50, and 100%. Use 10 replicas for exact steps.

---

## Repository structure

```
.
├── src/
│   ├── validation/        # cleaning rules, Pandera schema, drop report
│   ├── features/          # shared feature code, geo, rolling, freshness, Feast store
│   ├── training/          # baseline, LightGBM, metrics, MLflow registry
│   ├── evaluation/        # compare, error analysis, CI gate, promotion
│   └── serving/           # FastAPI app, live weather, prediction log
├── flows/                 # Prefect flows: retrain, drift, month state, refresh
├── monitoring/            # drift jobs, simulator, surge demo, alerts, Prometheus, Grafana
├── k8s/                   # k3d manifests, Argo Rollout, analysis template, demo driver
├── ui/                    # Streamlit demo UI (separate image, own requirements and tests)
├── scripts/               # data download, zone centroids, verify, UI zone list
├── data/
│   ├── raw/               # DVC-tracked inputs (not in git)
│   └── reference/         # zone_centroids.csv (committed, small)
├── docs/                  # RUNBOOK.md and handoff notes
├── tests/                 # pytest suite
├── .github/               # workflows, CODEOWNERS, metric reference files
├── dvc.yaml               # clean stage
├── docker-compose.yml     # local stack
├── Dockerfile             # serving image
└── pyproject.toml
```

The interfaces between components (cleaned-table schema, registered model name, `/predict` JSON, metric names, drift report format) are defined in `docs/contracts.md`, which the two owners keep out of the repository. Changing a contract requires approval from both owners.

---

## Testing and quality

| Check | What it covers |
|---|---|
| `pytest` | 290+ tests: cleaning rules, features and geo, skew test, training and registry, gate, promotion, API behavior, live weather (mocked HTTP), prediction log, drift calibration, surge demo, manifests, Dockerfile, DVC config |
| `pytest ui/tests` | 16 UI tests (Streamlit `AppTest`, mocked HTTP), run in their own CI job from a clean environment |
| `ruff check .` | Linting |
| `scripts/verify.py` | Six correctness checks on real data: no leakage columns, time split ordering, weather NaN rate under 1%, model beats baseline, seed reproducibility, unseen-zone prediction |
| CI metric gate | Fails a PR if the trained model's MAE regresses against `.github/metric_reference_real.json` |
| Manifest tests | Canary steps, metric names, fail-closed analysis conditions, read-only mount guards |

No test touches the real network; HTTP is mocked and a guard fixture makes any real call fail.

---

## Team and ownership

| Domain | [@vaishnavi7106](https://github.com/vaishnavi7106) | [@amizh-labs](https://github.com/amizh-labs) |
|---|---|---|
| Data | Feature engineering, weather join, Feast stores, skew test | Ingestion, cleaning, Pandera validation, DVC |
| ML | Evaluation, champion/challenger comparison, promotion | Training, MLflow registry, drift reports |
| MLOps | CI/CD, metric gate, image build | Prefect flows, retrain triggers |
| Deployment | k3d, Argo Rollouts canary, analysis template | FastAPI service, Docker, monitoring, dashboards |

---

## Data sources

- **NYC TLC Trip Record Data** (yellow taxi, 2026): https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **NYC TLC Taxi Zone Lookup and Shapefile**, used for zone validation and centroids.
- **Open-Meteo** (historical archive and forecast APIs): https://open-meteo.com

Trip data, zone files, and weather are public. The raw files and the zone shapefile are not committed; `scripts/download_data.py` rebuilds them.
