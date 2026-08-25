# Root Dockerfile - what Hugging Face Spaces, Render and Railway pick up by
# default. It is the PaaS image; docker/Dockerfile.paas is kept as the
# canonical copy for docker-compose and explicit builds.
#
# Local:  docker build -t oilspill . && docker run -p 7860:7860 oilspill
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal-dev gdal-bin libproj-dev libgl1 libglib2.0-0 \
        gcc g++ curl \
    && apt-get clean

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install global-land-mask \
    && apt-get purge -y gcc g++ && apt-get autoremove -y

COPY core/ core/
COPY ingest/ ingest/
COPY detect/ detect/
COPY drift/ drift/
COPY attribute/ attribute/
COPY decision/ decision/
COPY api/ api/
COPY ui/ ui/
COPY scripts/ scripts/
COPY configs/ configs/

# Fail the build loudly if a source file arrived empty. A truncated COPY
# produces a container that starts and exits with no error, which is far
# harder to diagnose than a build that refuses to finish.
RUN test -s api/main.py && test -s core/contracts.py && test -s ui/static/app.js \
    || (echo "FATAL: source files copied empty - check .dockerignore and disk space" && exit 1)

# Demo scenes and the documented-incident registry, baked in so a cold
# container has data on its first request. PaaS disks are ephemeral.
# Scenes are generated at 700 px rather than the 1400 px default. A free
# container has 512 MB, and analysing a 1400 px scene peaks past that - the
# health check survives but the first real request kills the worker, which
# surfaces as a 502 rather than an error. Quartering the pixel count keeps the
# whole pipeline inside the budget; the planted slicks stay clearly visible.
RUN mkdir -p data/demo_internal data/reference \
    && python scripts/make_demo_scene.py --size 700 \
    && python scripts/make_demo_scene.py --calm-wind --name CALM_WIND_DEMO \
         --bbox "74.20,9.05,74.80,9.65" --seed 21 --size 700 \
    && (python scripts/fetch_incidents.py || \
        echo "WARNING: incident registry unavailable at build time")

# 7860 is the Hugging Face Spaces convention; $PORT overrides it on Render,
# Railway and Fly, all of which inject their own.
ENV OILSPILL_CONFIG=configs/demo_synthetic.yaml \
    SERVE_UI=true \
    PORT=7860
EXPOSE 7860

# Invoked as a module rather than through the console script, which can carry
# a stale entry point on this base image.
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
