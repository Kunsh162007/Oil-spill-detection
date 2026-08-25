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

# The API-only dependency set. requirements.txt is the development list and
# pulls matplotlib, pandas and scikit-learn, which only the training and
# eval scripts use - on a 512 MB container that overhead alone can exhaust
# memory before a single request is served.
COPY requirements-api.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt \
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

# Scene manifests and their PRECOMPUTED analyses. Without this the image holds
# only the two scenes it generates below, and the deployed map shows a single
# detection where a local run shows dozens. .dockerignore keeps the imagery
# itself out - serving a cached analysis never touches the raster.
COPY data/ data/

# Fail the build loudly if a source file arrived empty. A truncated COPY
# produces a container that starts and exits with no error, which is far
# harder to diagnose than a build that refuses to finish.
RUN test -s api/main.py && test -s core/contracts.py && test -s ui/static/app.js \
    || (echo "FATAL: source files copied empty - check .dockerignore and disk space" && exit 1)

# The cached analyses are the deployed map's entire content. A .dockerignore
# edit that silently drops them yields a working service with one detection
# on it, which is far harder to notice than a build that stops here.
RUN n=$(ls data/precomputed/*.pkl 2>/dev/null | wc -l); m=$(cat data/live/*.json data/demo_finale/*.json 2>/dev/null | grep -c scene_id); echo "precomputed analyses: $n   scene manifests: $m"; test "$n" -ge 10 && test "$m" -ge 10 || (echo "FATAL: precomputed analyses missing - check .dockerignore data/ rules" && exit 1)

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

# Analyse every scene here, where memory is plentiful, and ship only the
# result. A 512 MB container cannot run the pipeline per request - the worker
# is OOM-killed mid-request, which surfaces as a 502 with no body - but it can
# comfortably deserialise a few hundred KB of polygons and scores.
# --skip-existing leaves the shipped analyses alone. Their imagery is not in
# the image, so re-analysing them would fail and overwrite a good cache; only
# the two scenes generated above still need computing.
RUN python scripts/precompute.py --skip-existing     || echo "WARNING: precompute failed; the container will analyse on demand and may exhaust memory"

# 7860 is the Hugging Face Spaces convention; $PORT overrides it on Render,
# Railway and Fly, all of which inject their own.
# ALLOW_LIVE_ANALYSIS=false: every scene in this image already has a cached
# analysis, and running the pipeline here would fail anyway - the coastline
# grid alone is 933 MB resident against a 512 MB limit. Refusing the request
# beats being OOM-killed and taking the whole map down.
ENV OILSPILL_CONFIG=configs/demo_synthetic.yaml \
    SERVE_UI=true \
    ALLOW_LIVE_ANALYSIS=false \
    PORT=7860
EXPOSE 7860

# Invoked as a module rather than through the console script, which can carry
# a stale entry point on this base image.
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
