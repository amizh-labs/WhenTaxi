# ETA MLOps Platform

End-to-end MLOps platform predicting NYC yellow taxi trip duration (ETA).
The platform (CI/CD, canary, drift, retraining) is the point; the model stays simple (LightGBM).

## Git and ownership rules
- Commit ONLY under the git identity configured on this machine. Never set another
  person's author name or email. Use `Co-authored-by:` trailers for paired work.
- Each teammate builds only their own packages on their own machine.
- Every PR is reviewed by the other teammate, who must be able to explain it.

| Area | Owner |
|---|---|
| `src/validation/`, `src/training/`, `flows/`, `src/serving/`, `monitoring/` | A: amizh (`amizh-labs`) |
| `src/features/`, `src/evaluation/`, `.github/`, `k8s/` | B: vaish (`vaishnavi7106`) |

Cross-package interfaces are defined in `docs/contracts.md`; change them only via a PR both approve.

## Constraints
- Zero-cost, open-source tools only. No AWS/GCP/Azure, no paid services.
- Windows machine, 24 GB RAM, Docker Desktop. Use pathlib; no bash-only scripts.
- Stack: DVC, Pandera, Feast, Prefect, MLflow, LightGBM, FastAPI, Docker, GitHub Actions,
  k3d + Argo Rollouts, Prometheus, Grafana, Evidently. Do not add tools outside this stack.
- Add tests for each stage. Explain what was built after each stage.

## Data (`data/raw`, tracked by DVC, never committed to git)
- `yellow_tripdata_2026-01/02/03.parquet`; Apr-Jul 2026 are held out as "new monthly data".
- `taxi_zone_lookup.csv`, `weather_2026_01_07.json` (Open-Meteo hourly, NYC local time).
- Columns used: tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID.
- Target: `duration_min = dropoff - pickup`.
- Known problems: pickups outside the file's month, zero/negative/huge durations, zone IDs
  not in the lookup. Cleaning drops these and writes a per-rule report.
- 2025+ files have an extra `cbd_congestion_fee` column. Never mix 2024 and 2026 data.

## Modeling rules
- Split by time (train Jan-Feb, validate Mar), never random.
- Do NOT use trip_distance, fare_amount, total_amount (leakage).
- Request-time features only: pickup zone, dropoff zone, hour, day of week, weekend, weather.
- Metrics: MAE (headline), RMSE, p90 absolute error; always compare to a zone-pair-mean baseline.
- Feature code lives in one module (`src/features/`) imported by both training and serving.
