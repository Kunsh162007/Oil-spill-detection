"""Pipeline control endpoints - the API of the pipeline itself.

The read endpoints in api/main.py answer "what did you find". These drive the
pipeline: fetch imagery for an area, analyse a scene, refresh the incident
registry. They exist so the frontend is a client of the pipeline rather than
a page bolted onto it, and so anything else (a scheduler, a notebook, another
service) can drive it the same way.

Everything long-running returns a job id immediately. A scene fetch takes
tens of seconds and a PaaS router will cut the connection long before it
finishes.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.config import REPO_ROOT

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


class FetchRequest(BaseModel):
    """Fetch Sentinel-1 scenes for an area. No credentials required."""

    bbox: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="min_lon, min_lat, max_lon, max_lat",
        examples=[[68.0, 8.0, 78.0, 20.0]],
    )
    days: float = Field(3.0, gt=0, le=30, description="search window, days")
    date: str | None = Field(
        None, description="centre on this date (YYYY-MM-DD) for archived incidents"
    )
    max_scenes: int = Field(2, ge=1, le=10)
    resolution_m: float = Field(80.0, ge=10.0, le=500.0)
    label: str | None = None


class AnalyseRequest(BaseModel):
    scene_id: str
    force: bool = Field(False, description="re-run even if already analysed")


def _validate_bbox(bbox: list[float]) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (-180 <= min_lon < max_lon <= 180):
        raise HTTPException(400, f"bad longitude range: {min_lon}..{max_lon}")
    if not (-90 <= min_lat < max_lat <= 90):
        raise HTTPException(400, f"bad latitude range: {min_lat}..{max_lat}")
    # A whole-hemisphere request would fetch for hours and fill the disk.
    if (max_lon - min_lon) > 25 or (max_lat - min_lat) > 25:
        raise HTTPException(
            400,
            "bbox is too large (max 25 degrees per side). Sentinel-1 scenes are "
            "~250 km wide; request a region, not an ocean.",
        )
    return min_lon, min_lat, max_lon, max_lat


def _state() -> dict[str, Any]:
    from api.main import _state as app_state

    return app_state


@router.post("/fetch", status_code=202)
def fetch_scenes(request: FetchRequest):
    """Fetch Sentinel-1 scenes for an area. Returns a job id.

    Search uses the open CDSE catalogue; pixels come from the AWS Open Data
    mirror. Neither needs an account. Scenes are read windowed, so a request
    costs tens of megabytes rather than gigabytes.
    """
    _validate_bbox(request.bbox)
    store = _state().get("jobs")
    if store is None:
        raise HTTPException(503, "service still starting")

    job = store.create("fetch", request.model_dump())

    def work(j) -> dict[str, Any]:
        j.progress = "searching the catalogue"
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "fetch_aws.py"),
            "--bbox", ",".join(str(v) for v in request.bbox),
            "--days", str(request.days),
            "--max-scenes", str(request.max_scenes),
            "--resolution", str(request.resolution_m),
            "--out-dir", "data/live",
            "--config", "configs/live.yaml",
        ]
        if request.date:
            cmd += ["--date", request.date]
        if request.label:
            cmd += ["--name", request.label]

        j.progress = "reading scene windows from the mirror"
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(
                (proc.stderr or proc.stdout or "fetch failed").strip()[-500:]
            )

        # New manifests only become visible once the store rescans.
        api_store = _state().get("store")
        found = api_store.discover() if api_store else []
        j.progress = "done"
        return {
            "scenes_available": len(found),
            "output": proc.stdout.strip()[-1200:],
        }

    store.run(job, work)
    return {"job_id": job.job_id, "state": job.state,
            "poll": f"/api/pipeline/jobs/{job.job_id}"}


@router.post("/analyse", status_code=202)
def analyse_scene(request: AnalyseRequest):
    """Run the full detection and attribution pipeline on one scene."""
    state = _state()
    store, jobs = state.get("store"), state.get("jobs")
    if store is None or jobs is None:
        raise HTTPException(503, "service still starting")
    if request.scene_id not in store.manifests:
        raise HTTPException(404, f"unknown scene: {request.scene_id}")

    job = jobs.create("analyse", request.model_dump())

    def work(j) -> dict[str, Any]:
        j.progress = "ingest, detect, drift, attribute"
        analysis = store.get(request.scene_id, force=request.force)
        j.progress = "done"
        return {
            "scene_id": request.scene_id,
            "regions": analysis.stats.get("n_regions"),
            "confirmed": analysis.stats.get("n_confirmed"),
            "rejected": analysis.stats.get("n_rejected"),
            "timings": analysis.timings,
            "backend": analysis.stats.get("segmentation_backend"),
        }

    jobs.run(job, work)
    return {"job_id": job.job_id, "state": job.state,
            "poll": f"/api/pipeline/jobs/{job.job_id}"}


@router.post("/refresh-incidents", status_code=202)
def refresh_incidents():
    """Re-download the documented-incident registry."""
    jobs = _state().get("jobs")
    if jobs is None:
        raise HTTPException(503, "service still starting")
    job = jobs.create("refresh-incidents", {})

    def work(j) -> dict[str, Any]:
        j.progress = "downloading NOAA IncidentNews"
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "fetch_incidents.py"), "--force"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "refresh failed").strip()[-400:])
        state = _state()
        store = state.get("store")
        if store is not None:
            store.registry = store._load_registry(state["config"])
        return {"output": proc.stdout.strip()[-800:]}

    jobs.run(job, work)
    return {"job_id": job.job_id, "state": job.state,
            "poll": f"/api/pipeline/jobs/{job.job_id}"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    jobs = _state().get("jobs")
    if jobs is None:
        raise HTTPException(503, "service still starting")
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    return job.to_dict()


@router.get("/jobs")
def list_jobs(limit: int = Query(25, ge=1, le=200)):
    jobs = _state().get("jobs")
    if jobs is None:
        raise HTTPException(503, "service still starting")
    return {"jobs": [j.to_dict() for j in jobs.list(limit)]}


@router.get("/capabilities")
def capabilities():
    """What this deployment can actually do, and what the data timeliness is.

    Exists so a client does not have to guess. A frontend can disable the
    fetch control when the deployment has no writable disk, and can state the
    latency honestly without hardcoding it.
    """
    from drift.runner import opendrift_available

    state = _state()
    config = state.get("config")
    store = state.get("store")

    writable = True
    try:
        probe = REPO_ROOT / "data" / ".write-probe"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        writable = False

    return {
        "can_fetch_imagery": writable,
        "can_analyse": True,
        "storage_writable": writable,
        "scenes_loaded": len(store.manifests) if store else 0,
        "incident_registry": len(store.registry) if store and store.registry else 0,
        "backends": {
            "drift": "opendrift-openoil" if opendrift_available() else "analytical-advection",
            "wind": config.get("wind.source") if config else None,
            "currents": config.get("drift.currents_source") if config else None,
        },
        "timeliness": {
            "imagery_lag_hours": "3-24",
            "ais_lag_hours": "~72",
            "active_window_hours": 72,
            "statement": (
                "Near-real-time, not live. Sentinel-1 reaches public mirrors "
                "3-24 h after acquisition and free AIS lags about 72 h. "
                "Imagery is fetched on demand and cached; only analysed scenes "
                "are stored."
            ),
        },
        "data_sources": {
            "imagery_search": "Copernicus Data Space catalogue (open, no account)",
            "imagery_pixels": "AWS Open Data sentinel-s1-l1c (open, no account)",
            "incidents": "NOAA IncidentNews + curated world catalogue",
            "ais": "Global Fishing Watch (token required)",
        },
    }
