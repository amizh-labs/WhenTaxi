# Contracts

Interfaces between packages. Change only via a PR approved by both owners.
Items marked TBD are filled in by the owning package's first PR.

## 1. Cleaned table schema (A produces, B consumes)
Parquet, one row per valid trip:

| Column | Type | Notes |
|---|---|---|
| `pickup_ts` | datetime64[ns] | NYC local time, within the file's month |
| `PULocationID` | int | present in zone lookup |
| `DOLocationID` | int | present in zone lookup |
| `duration_min` | float | target; > 0 and below the configured max |
| `month` | str | `YYYY-MM` |

## 2. Registered model (A registers; A's API and B's promotion logic use it)
- MLflow registered model name: `eta-duration` (TBD confirm)
- Load: `mlflow.pyfunc.load_model("models:/eta-duration@champion")`
- Alias `champion` = serving model; `challenger` = candidate under evaluation.

## 3. `/predict` (A's API; B's canary and tests call it)
Request:
```json
{"pickup_zone": 161, "dropoff_zone": 237, "pickup_ts": "2026-03-10T08:30:00"}
```
Response:
```json
{"eta_minutes": 14.2, "model_version": "3"}
```

## 4. Prometheus metric names (A exports; B's Argo AnalysisTemplate queries)
- `eta_requests_total{status}` counter
- `eta_request_latency_seconds` histogram
- `eta_prediction_minutes` histogram
- (TBD) error/abs-error metric once ground truth feedback exists

## 5. Drift output (A's Evidently job writes; A's Prefect flow and B's promotion read)
`monitoring/reports/drift_<YYYY-MM>.json`:
```json
{"month": "2026-04", "dataset_drift": true, "drifted_columns": ["temperature_2m"], "share_drifted": 0.2}
```

## 6. CONTRACT CHANGE (proposed, needs both approvals): geography features in the registered model
Touches section 2 (registered model) and the `/predict` input path. Needs a PR both teammates approve.
- New shared columns in `features.GEO_COLUMNS`: `distance_km`, `bearing_deg`, `same_zone`, `borough_pair`
  (pickup borough code * 10 + dropoff borough code), `airport_flag`. All derive from the two zone ids only,
  via `features.geo_features`, reading `data/reference/zone_centroids.csv` (no geospatial library at runtime).
- A model "needs geo" when all `GEO_COLUMNS` are in its `feature_columns`; then `build_features(..., geo=True)`
  must be used (the API does this automatically via `LoadedModel.needs_geo`). Request body is unchanged.
- Zones without a centroid (264, 265, unknown ids) give NaN `distance_km`/`bearing_deg`, never an error.
- A model trained with `log_target` (log1p of the duration) is registered with run param `log_target=True`;
  `LoadedModel.predict` applies expm1, so consumers still receive minutes.
- Geo features are not in Feast on purpose (static, pure function of the zone pair: no skew possible).

## 7. CONTRACT CHANGE (proposed, needs both owners' approval): live weather and `weather_source` in `/predict`
Touches section 3. The request body is unchanged (optional `weather` override already existed). The response gains one field:
```json
{"eta_minutes": 14.2, "model_version": "3", "weather_source": "live"}
```
- `weather_source` is one of `request`, `live`, `archive`, `missing`. Source order per request: values supplied in
  the request's `weather` > live Open-Meteo forecast > the archive file > missing (NaN weather, no error).
  `request` is reported whenever any weather field was supplied; fields left out are filled from the rest of the chain.
- Live = Open-Meteo `/v1/forecast` with `hourly=temperature_2m,precipitation,snowfall,wind_speed_10m`,
  `timezone=America/New_York`, default units (degC, mm, cm, km/h), the row of the pickup hour. Never the `current` block.
  2 s timeout, one call per clock hour (cached), a failure is remembered for 60 s. Only pickups from yesterday to
  2 days ahead are looked up live; all other timestamps use the archive file.
- `ETA_LIVE_WEATHER=0` disables the live call. Consumers must tolerate (ignore) the new field.

## 8. CONTRACT CHANGE (proposed, needs both owners' approval): drift report on real requests
Touches section 5. `monitoring/prediction_drift.py` (log-based drift) writes `drift_report.json` with:
```json
{"month": "2026-11-18", "dataset_drift": true, "drifted_columns": ["distance_km"], "share_drifted": 0.3333,
 "context_drift": {"columns": ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m"],
                   "drifted_columns": ["snowfall"], "share_drifted": 0.25},
 "data_quality": {"share_missing_weather": 0.02, "alert": false},
 "status": "ok", "n_rows": 500, "window_start": "...", "window_end": "..."}
```
- Only `hour`, `dow` and `distance_km` can set `dataset_drift` (true when at least one drifts; KS test p < 0.003).
  `drifted_columns` / `share_drifted` therefore refer to those three trigger columns only.
- The weather columns are context: listed in `context_drift`, never in `dataset_drift`.
- Missing weather is reported in `data_quality` (alert above 20% of requests); it never sets `dataset_drift`.
- `status` is `"insufficient data"` (and `dataset_drift` false) below 200 logged requests or with no log.
- The older keys `share_drifted_features` and `drifted_features` stay, so `flows/retrain.py` is unaffected.

## 9. CONTRACT CHANGE (proposed, needs both owners' approval): freshness metrics (section 4)
The API's `/metrics` gains two gauges (existing names unchanged):
- `eta_feature_age_seconds`: seconds since the Feast online store was materialized, read from `<ETA_FEAST_ROOT>/materialized.json`
  (written by `features.store.materialize`). `-1` means unknown (no or unreadable file).
- `eta_feature_age_alert_threshold_seconds`: from env `ETA_FEATURE_AGE_ALERT_SECONDS` (default 604800).
Prometheus rule `EtaFeaturesStale` (`monitoring/alerts.yml`, mirrored in `k8s/prometheus.yaml`): `eta_feature_age_seconds > eta_feature_age_alert_threshold_seconds`
for 1m. No Alertmanager; alerts are visible in the Prometheus UI.
