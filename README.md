# ETA MLOps Platform

(Full README still to be written; this file currently documents the data only.)

## Data

- `data/raw/` (DVC-tracked, never in git): TLC yellow-taxi trip files, `taxi_zone_lookup.csv`, Open-Meteo weather.
- `data/reference/zone_centroids.csv` (committed, small): one lat/lon centroid per taxi zone, used by
  `features.geo` for distance and bearing features. Zones 264/265 ("Unknown") have no polygon and are absent.
  - Source: NYC TLC Taxi Zone Shapefile, `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip`
  - Downloaded: 2026-11-18. The shapefile itself is not committed.
  - Rebuild (rare, needs the optional extra): `pip install -e ".[geo]"` then
    `python scripts/build_zone_centroids.py --download <empty dir>`.

## Prediction log (live / Compose deployment only)

`/predict` can append every call to a SQLite file so real requests can be compared with the training data.

- **Off by default.** Enable with `ETA_PREDICTION_LOG=<path>` (for example `data/prediction_log.sqlite`).
  Optional `ETA_PREDICTION_LOG_MAX_ROWS` (default 100000) keeps only the newest rows.
- **Single-instance only.** The log is meant for the single-container Compose/live deployment. The k3d cluster demo runs
  5-10 API replicas that would fight over one SQLite file, so it leaves the log off and keeps only Prometheus metrics.
- **Columns:** `ts` (UTC), `pickup_zone`, `dropoff_zone`, `pickup_ts`, the four weather values the model used,
  `weather_source` (`request` / `live` / `archive` / `missing`), `eta_minutes`, `model_version`.
- **Cannot hurt `/predict`.** The request only puts a record on a bounded in-memory queue; a background thread writes it
  (WAL mode). A full queue drops records and any write error is swallowed (both are counted); the response is unchanged.
- **Drift on real requests:** `python -m monitoring.prediction_drift --log <path>` compares the logged window with a
  20k-row sample of Jan-Feb 2026 on hour, day of week, the four weather columns and the geo distance between zones
  (raw zone ids are not compared: too many categories and too noisy at a few hundred requests). It writes
  `monitoring/reports/drift_report.json` (contract format, `docs/contracts.md` section 5) and an HTML report.
  With fewer than 200 logged rows it reports `"status": "insufficient data"` and never flags drift.
  Weather in November will honestly differ from January-February, so expect temperature to drift on a live log.

**Drift check caveats.**
- The drift check assumes representative traffic: a log of a few test requests (or one rider type) will look "drifted"
  against the Jan-Feb reference. Demo drift runs therefore replay real trips (`monitoring/simulator.py`, or the surge demo below),
  not hand-sent requests.
- November weather will differ from the Jan-Feb reference. That is why the four weather columns are reported as
  context (`context_drift`) but can never set `dataset_drift`; only hour, day of week and zone distance can.
- Missing weather is a data-quality signal, not drift: the report's `data_quality.share_missing_weather` carries an
  `alert` flag above 20%.
- Compose runs the log on a named volume (`prediction-log`). The daily check is a Prefect flow:
  `python -m flows.drift_flow` (once) or `python -m flows.drift_flow --serve` (daily at 06:00); copy the log out of the
  volume first with `docker compose cp api:/app/log/predictions.sqlite data/prediction_log.sqlite`.

## Data versioning (DVC)

- **What is tracked.** `data/raw` (trip files, zone lookup, weather) is tracked by DVC through `data/raw.dvc`; git
  only holds that small pointer. `data/processed/trips_clean.parquet` is produced by the `clean` stage in `dvc.yaml`,
  with the drop report `data/processed/data_quality_report.json` as its metric file.
- **Track or update the raw data:** `dvc add data/raw`, then commit the changed `data/raw.dvc`.
- **Local remote.** `.dvc/config` points the remote `localremote` at a folder on one machine (`/home/user1/dvc-remote`);
  `dvc push` / `dvc pull` copy data to and from it. It is free and needs no account, but it is not shared between machines.
- **Rebuild the cleaned data:** `dvc repro` runs the `clean` stage. A second run prints "Data and pipelines are up
  to date" and does nothing, until the raw data or `src/validation/clean.py` / `schema.py` change.
- **Fresh clone** (no DVC cache and no access to the other machine's remote): run `python scripts/download_data.py`
  to fetch the raw files, then `dvc add data/raw` to track them, then `dvc repro`. Identical files give an identical hash,
  so `data/raw.dvc` stays unchanged.

## Evening-rush surge demo (Compose + WSL2)

`python -m monitoring.surge_demo` shows the platform **detecting a shift in traffic and retraining a challenger**. It does
**not** show the model adapting: the challenger is trained on January-March and validated on April, and has never seen
the surge. Commands: `docs/RUNBOOK.md`, section "The evening-rush surge demo".

It replays cleaned, held-out April 2026 trips (`data/holdout/yellow_tripdata_2026-04.parquet`, outside `data/raw`)
through `/predict` on the Compose stack, in two windows of N requests (default 2000):

| Window | Trips | Ground truth for MAE |
|---|---|---|
| baseline | April as it is | the true durations |
| surge | resampled toward 16:00-21:59 (85%) with 30% airport trips, weekday mix kept | the true durations **x1.4, SYNTHETIC** |

The check is split in two on purpose:

- **Drift comes from the API.** It uses only what the prediction log holds (`pickup_ts` and the two zones), so it is sound
  for replayed history. Expected: the baseline shows no drift; the surge drifts on `hour` and `distance_km` only among the
  trigger columns; the weather columns are context only.
- **MAE is computed offline**, with the same code as the monthly evaluation (`evaluation.compare.evaluate`) and rolling
  features computed point-in-time over the last 27 hours of March plus April. The champion is only read. The surge MAE is
  scored against the **synthetic** inflated durations, so its rise is built in; the point is that the flag fires. The flag
  is "MAE more than 10% over the baseline window"; that 10% is this demo's own parameter (`--mae-threshold`), not the CI
  gate (+2%) or the promotion rules.
- **Retrain.** When drift or the MAE flag fires, the existing retrain flow runs against a **throwaway** MLflow SQLite store
  and throwaway processed and Feast folders under `data/processed/surge_demo/`, on a row sample by default
  (`--retrain-rows`, or `--full-retrain`). The real registry, the real `champion` alias, `data/processed` and `data/feast`
  are never written. Result: a summary table and `data/processed/surge_demo_report.json`.

## Limitations

- **`/predict` reads its rolling features from one materialized snapshot and ignores `pickup_ts`.** The features
  `avg_duration_1h`, `avg_duration_24h` and `trip_count_24h` come from the Feast online store as of the last
  `features.store.materialize` run, whatever timestamp the request carries. So predictions for historical timestamps use
  the snapshot's traffic levels, not the traffic at that time. Measured on 20,000 March trips against the snapshot as of
  2026-04-01: `avg_duration_24h` differs from the point-in-time value by about 2 minutes on average, and `avg_duration_1h`
  is present for 62% of requests against 74% point-in-time. These three features carry about 72% of the model's importance
  (by gain), so this matters. For live traffic the snapshot is meant to be fresh; the `EtaFeaturesStale` alert exists to
  expose staleness (and `EtaFeaturesUnknown` covers a store that was never materialized). Live use therefore needs frequent
  re-materialization, or a request-time override of the rolling features (see "What I would do next").
- **The model is largely weather-insensitive.** The weather columns account for about 1.7 percent of the feature
  importance, so predictions barely move with the weather. They appear in the drift report as context only.
- **Replay MAE is offline on purpose** (see the surge demo): scoring replayed trips through `/predict` would mix in the
  stale-snapshot effect above.
- **Thresholds are not tuned to this data.** The day-of-week drift effect floor (0.10) sits close to the month's own
  weekday-mix difference from January-February (about 0.08-0.09 on March), so a small plain random window can cross it by
  sampling noise alone. The surge demo draws its windows with the month's weekday mix for that reason.

## What I would do next

- **Request-time override of the rolling features** (a request field, like the existing weather override), so a replay or a
  what-if request can carry point-in-time values. This changes the API contract (`docs/contracts.md`), so it needs both
  owners' approval.
- Re-materialize the Feast store on a schedule and alert on its age (the alert exists; the schedule does not).
- More winter data and a weather-aware model, only if weather sensitivity is wanted.

