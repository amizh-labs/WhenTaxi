FROM python:3.11-slim

# LightGBM needs the OpenMP runtime, which the slim base image does not include.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# Static zone centroids for the geo features (committed CSV, ~15 KB). pyshp/pyproj (the `geo` extra) are
# only for the one-time scripts/build_zone_centroids.py and are deliberately NOT installed here.
COPY data/reference ./data/reference
RUN pip install --no-cache-dir ".[serving]"

# Raw data is DVC-tracked and never in git, so it is not baked into the image: the weather file
# and the Feast online store are mounted at run time (see docker-compose.yml).
ENV ETA_CENTROIDS_PATH=/app/data/reference/zone_centroids.csv \
    ETA_WEATHER_PATH=/app/data/raw/weather_2026_01_07.json \
    MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000
CMD ["uvicorn", "serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
