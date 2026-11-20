"""FastAPI service: POST /predict, GET /health, GET /metrics (Prometheus).

Contract: docs/contracts.md sections 3-4. Features are built with the SAME build_features
and zone category list used in training. Weather for the pickup hour comes from the
Open-Meteo file; a request may override weather values (used by the snowstorm demo).
"""
import contextlib
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from features import WEATHER_COLUMNS, build_features, feature_age_seconds, load_weather
from features.freshness import MISSING_AGE
from serving.live_weather import LiveWeather
from serving.prediction_log import default_prediction_log

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGE_ALERT = 7 * 24 * 3600
DEFAULT_WEATHER = ROOT / "data" / "raw" / "weather_2026_01_07.json"


class WeatherOverride(BaseModel):
    temperature_2m: float | None = None
    precipitation: float | None = None
    snowfall: float | None = None
    wind_speed_10m: float | None = None


class PredictRequest(BaseModel):
    pickup_zone: int = Field(ge=1)
    dropoff_zone: int = Field(ge=1)
    pickup_ts: datetime  # ISO 8601, NYC local time
    weather: WeatherOverride | None = None  # optional extension; ignored when absent


class PredictResponse(BaseModel):
    eta_minutes: float
    model_version: str
    weather_source: str  # request | live | archive | missing (docs/contracts.md section 3)


class Metrics:
    """Prometheus metrics with the names fixed by the contract (own registry per app)."""

    def __init__(self, feast_root: Path | None = None):
        self.registry = CollectorRegistry()
        self.requests = Counter("eta_requests_total", "Prediction requests", ["status"], registry=self.registry)
        self.latency = Histogram("eta_request_latency_seconds", "Prediction latency", registry=self.registry)
        self.prediction = Histogram(
            "eta_prediction_minutes", "Predicted ETA in minutes",
            buckets=(2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180), registry=self.registry)
        # Freshness of the rolling features: seconds since the Feast store was materialized (read from
        # <feast root>/materialized.json at every scrape); -1 = unknown (no file). The alert fires when the age
        # exceeds the threshold gauge, which comes from ETA_FEATURE_AGE_ALERT_SECONDS (default 7 days).
        root = feast_root or Path(os.environ.get("ETA_FEAST_ROOT", ROOT / "data" / "feast"))
        Gauge("eta_feature_age_seconds", "Seconds since the Feast online store was materialized (-1 = unknown)",
              registry=self.registry).set_function(lambda: MISSING_AGE if (a := feature_age_seconds(root)) is None else a)
        Gauge("eta_feature_age_alert_threshold_seconds", "Feature age above which EtaFeaturesStale fires",
              registry=self.registry).set(float(os.environ.get("ETA_FEATURE_AGE_ALERT_SECONDS", DEFAULT_AGE_ALERT)))


def default_loader():
    from training import load_model

    return load_model(os.environ.get("ETA_MODEL_ALIAS", "champion"))


def default_rolling_provider():
    """Feast online store under ETA_FEAST_ROOT (default data/feast); None if it was never built."""
    root = Path(os.environ.get("ETA_FEAST_ROOT", ROOT / "data" / "feast"))
    if not (root / "registry.db").exists():
        return None
    from features.store import FeastRollingProvider  # lazy: only models that need it pay for feast

    return FeastRollingProvider(root)


def default_live_weather():
    """Open-Meteo forecast lookup; ETA_LIVE_WEATHER=0 turns it off (offline demos, tests)."""
    return None if os.environ.get("ETA_LIVE_WEATHER", "1") == "0" else LiveWeather()


def create_app(model_loader=default_loader, weather_path: Path | None = None,
               rolling_provider=default_rolling_provider, live_weather=default_live_weather,
               prediction_log=default_prediction_log) -> FastAPI:
    """model_loader() -> object with .predict(features_df), .zone_categories, .version.

    rolling_provider() -> object with .get(pickup_zone, dropoff_zone) -> dict of rolling features,
    or None. Used only when the model was trained with rolling features; without a provider the
    features are NaN / count 0 (the model degrades gracefully, as for a never-seen zone pair).

    prediction_log() -> PredictionLog or None (ETA_PREDICTION_LOG=<path> turns it on; off by default). Logging is
    fire-and-forget on a background thread and can never fail or slow a request.

    live_weather() -> object with .lookup(pickup_ts) -> dict | None, or None. Weather source order per request:
    request-supplied values, live forecast, archive file, missing (NaN). The response says which one was used.
    """
    metrics = Metrics()
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state["model"] = model_loader()
        state["rolling"] = rolling_provider() if rolling_provider else None
        state["live"] = live_weather() if live_weather else None
        weather = Path(os.environ.get("ETA_WEATHER_PATH", weather_path or DEFAULT_WEATHER))
        state["weather"] = load_weather(weather) if weather.exists() else None
        state["log"] = prediction_log() if prediction_log else None
        yield
        if state["log"] is not None:
            state["log"].close()

    app = FastAPI(title="ETA service", lifespan=lifespan)

    @app.middleware("http")
    async def record(request: Request, call_next):
        if request.url.path != "/predict":
            return await call_next(request)
        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            metrics.latency.observe(time.perf_counter() - start)
            metrics.requests.labels(status=status).inc()

    @app.get("/health")
    def health():
        model = state.get("model")
        return {"status": "ok", "model_version": getattr(model, "version", None)}

    @app.get("/metrics")
    def prometheus():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        # Deliberate faults for canary/rollback demos and tests only (ETA_FAULT is never set in prod):
        #   errors -> every request fails with 500; slow -> +0.8s latency; bias -> predictions x3.
        fault = os.environ.get("ETA_FAULT", "")
        if fault == "errors":
            raise HTTPException(status_code=500, detail="injected fault")
        if fault == "slow":
            time.sleep(0.8)
        model = state["model"]
        archive = state["weather"]
        pickup = pd.Timestamp(req.pickup_ts).tz_localize(None)
        trips = pd.DataFrame({"pickup_ts": [pickup], "PULocationID": [req.pickup_zone],
                              "DOLocationID": [req.dropoff_zone]})
        supplied = req.weather.model_dump(exclude_none=True) if req.weather else {}
        weather_source = "request" if supplied else None
        weather = pd.DataFrame({"weather_hour": pd.to_datetime([]), **{c: [] for c in WEATHER_COLUMNS}})
        if not supplied or len(supplied) < len(WEATHER_COLUMNS):  # fill whatever the request left out
            live = state["live"].lookup(pickup) if state.get("live") else None
            hour = pickup.floor("h")
            if live is not None:
                weather = pd.DataFrame({"weather_hour": [hour], **{c: [v] for c, v in live.items()}})
                weather_source = weather_source or "live"
            elif archive is not None and (archive["weather_hour"] == hour).any():
                weather, weather_source = archive, weather_source or "archive"
            else:
                weather_source = weather_source or "missing"
        rolling = None
        if getattr(model, "needs_rolling", False):
            provider = state["rolling"]
            values = (provider.get(req.pickup_zone, req.dropoff_zone) if provider else
                      {"avg_duration_1h": float("nan"), "avg_duration_24h": float("nan"), "trip_count_24h": 0.0})
            rolling = pd.DataFrame([values], index=trips.index)
        features = build_features(trips, weather, model.zone_categories, rolling, geo=getattr(model, "needs_geo", False))
        for name, value in supplied.items():
            features[name] = value
        eta = float(model.predict(features).iloc[0])
        if fault == "bias":
            eta *= 3.0
        metrics.prediction.observe(eta)
        if state.get("log") is not None:
            with contextlib.suppress(Exception):  # fail open: a logging problem must never change the response
                state["log"].log(pickup_zone=req.pickup_zone, dropoff_zone=req.dropoff_zone,
                                 pickup_ts=pickup.isoformat(), weather=features[WEATHER_COLUMNS].iloc[0].to_dict(),
                                 weather_source=weather_source, eta_minutes=eta, model_version=model.version)
        return PredictResponse(eta_minutes=eta, model_version=str(model.version), weather_source=weather_source)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    return app
