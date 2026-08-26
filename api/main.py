"""FastAPI backend.

Endpoints the map needs:

    GET  /api/health                       service + backend status
    GET  /api/scenes                       every known scene
    GET  /api/slicks                       world index of confirmed slicks
    GET  /api/scenes/{id}                  one scene as GeoJSON
    POST /api/scenes/{id}/analyse          force re-analysis
    GET  /api/slicks/{candidate_id}        full detail: origin, drift, vessels
    GET  /api/slicks/{candidate_id}/backtrace   drift animation frames
    GET  /api/cerulean                     external reference slicks

Every response that names a vessel also carries the disclaimer. That is
deliberate: the caveat should be impossible to drop by accident.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import REPO_ROOT, load_config
from api.serialize import (attribution_detail, refresh_ages, scene_collection,
                           world_index)
from api.store import AnalysisStore, live_analysis_allowed
from decision.rank import DISCLAIMER

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("api")

CONFIG_PATH = os.environ.get("OILSPILL_CONFIG", "configs/demo_synthetic.yaml")
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from api.jobs import JobStore

    config = load_config(CONFIG_PATH)
    store = AnalysisStore(config)
    manifests = store.discover()
    _state["store"] = store
    _state["config"] = config
    _state["jobs"] = JobStore()
    log.info(
        "API ready with %d scene(s): %s",
        len(manifests), ", ".join(m["scene_id"] for m in manifests) or "none",
    )
    if not manifests:
        log.warning(
            "No scene manifests found. Run: python scripts/make_demo_scene.py"
        )
    yield
    _state.clear()


app = FastAPI(
    title="Oil Spill Detection & Vessel Attribution",
    description=(
        "SIH 2026 / SIH26143. Near-real-time SAR oil-spill detection with "
        "physics-based look-alike rejection, backward drift to origin, and "
        "ranked vessel attribution. " + DISCLAIMER
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Origins allowed to call this API. The frontend is a separate deployment, so
# it must be named here - a wildcard is acceptable for a local demo but not
# for a public service.
_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
ALLOWED_ORIGINS = (
    [o.strip() for o in _origins_env.split(",") if o.strip()] if _origins_env else ["*"]
)
if ALLOWED_ORIGINS == ["*"]:
    log.warning(
        "CORS_ORIGINS is unset: allowing any origin. Set it to your frontend "
        "URL before exposing this service publicly."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

from api.pipeline_routes import router as pipeline_router  # noqa: E402

app.include_router(pipeline_router)


def get_store() -> AnalysisStore:
    store = _state.get("store")
    if store is None:
        raise HTTPException(503, "Service still starting")
    return store


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Service status, including which backends are real and which are stubs."""
    store = _state.get("store")
    config = _state.get("config")
    from drift.runner import opendrift_available

    # Deliberately does NOT import torch. Importing it costs ~160 MB resident,
    # and a health check is the one request that must stay cheap - platforms
    # poll it continuously, and it has to answer while the service is under
    # memory pressure. If an analysis has already loaded torch we report what
    # it found; otherwise CUDA is simply unknown from here.
    torch_cuda = False
    torch_module = sys.modules.get("torch")
    if torch_module is not None:
        try:
            torch_cuda = bool(torch_module.cuda.is_available())
        except Exception:  # pragma: no cover - driver problems are not our concern here
            torch_cuda = False

    checkpoint = REPO_ROOT / str(config.get("detect.checkpoint", "models/stage_b.pt")) if config else None

    return {
        "status": "ok" if store else "starting",
        "scenes": len(store.manifests) if store else 0,
        "analysed": store.analysed_ids() if store else [],
        "backends": {
            "segmentation": (
                "trained-unet" if checkpoint and checkpoint.exists()
                else "classical-dark-patch (no checkpoint - run scripts/train.py)"
            ),
            "drift": "opendrift-openoil" if opendrift_available() else "analytical-advection",
            "cuda": torch_cuda,
            "wind": (config.get("wind.source", "synthetic") if config else None),
            "currents": (config.get("drift.currents_source", "synthetic") if config else None),
        },
        "memory": _memory_report(),
        "disclaimer": DISCLAIMER,
        "timeliness": "near-real-time (imagery 3-24 h; free AIS ~72 h)",
    }


def _memory_report() -> dict[str, Any]:
    """Resident and peak memory, so a container's real usage is observable.

    Free tiers report an OOM kill and nothing else - no traceback, no line
    number, and the process that could have told you is gone. Publishing the
    numbers on the health endpoint means the next memory question is answered
    by measurement rather than by inference from a crash.
    """
    report: dict[str, Any] = {"limit_hint_mb": 512}
    try:
        import resource

        # ru_maxrss is KB on Linux, bytes on macOS.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_mb"] = round(peak / (1024 if sys.platform != "darwin" else 1024 ** 2), 1)
    except (ImportError, OSError):
        pass
    try:
        # Linux only, and the number that actually matters: current RSS.
        with open("/proc/self/statm", "r", encoding="utf-8") as fh:
            pages = int(fh.read().split()[1])
        report["rss_mb"] = round(pages * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except (OSError, ValueError, AttributeError):
        pass
    report["heavy_imports"] = sorted(
        m for m in ("torch", "global_land_mask", "rasterio", "skimage",
                    "scipy", "pandas", "sklearn", "matplotlib")
        if m in sys.modules
    )
    return report


@app.get("/api/scenes")
def list_scenes() -> dict[str, Any]:
    """Every scene manifest on disk, analysed or not."""
    store = get_store()
    scenes = []
    for scene_id, data in store.manifests.items():
        scenes.append({
            "scene_id": scene_id,
            "acquired_at": data.get("acquired_at"),
            "bbox": data.get("bbox"),
            "synthetic": bool(data.get("SYNTHETIC")),
            "note": data.get("note"),
            "analysed": scene_id in store.analysed_ids(),
            "has_ais": bool(data.get("ais_path")),
        })
    return {"scenes": scenes, "count": len(scenes)}


WORLD_INDEX_CACHE = "data/precomputed/world_index.json"


def _cached_world_index() -> dict | None:
    """The world index as built at image build time, if it was baked in.

    Holding every scene's analysis in memory at once is what a small container
    cannot afford - not the pipeline, the RESULTS. Thirteen scenes deserialised
    together crash a 512 MB worker, and the map needs all of them on its first
    request. Reading one JSON file instead keeps that cost flat.
    """
    from core.config import resolve_path

    path = resolve_path(WORLD_INDEX_CACHE)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Cached world index unusable (%s); building it live", exc)
        return None


@app.get("/api/slicks")
def all_slicks(analyse: bool = Query(True, description="analyse any scene not yet processed")):
    """World index: confirmed slicks across every scene. Feeds the main map."""
    store = get_store()

    cached = _cached_world_index()
    if cached is not None:
        # Ages are recomputed per request; everything else is fixed at build.
        return JSONResponse(refresh_ages(cached))

    analyses = []
    errors: dict[str, str] = {}

    for scene_id in store.manifests:
        try:
            analyses.append(store.get(scene_id))
        except Exception as exc:
            # One broken scene must not empty the whole map.
            log.error("Analysis failed for %s: %s", scene_id, exc)
            errors[scene_id] = str(exc)

    payload = world_index(analyses)
    if errors:
        payload["meta"]["errors"] = errors
    return JSONResponse(payload)


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str):
    """One scene as GeoJSON, including rejected look-alikes and their reasons."""
    store = get_store()
    try:
        analysis = store.get(scene_id)
    except KeyError:
        raise HTTPException(404, f"Unknown scene: {scene_id}")
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed for {scene_id}: {exc}")
    return JSONResponse(scene_collection(analysis))


@app.post("/api/scenes/{scene_id}/analyse")
def reanalyse(scene_id: str):
    """Force a re-run, ignoring the cached analysis."""
    store = get_store()
    if not live_analysis_allowed():
        # 503 rather than a crash: refusing is far better than being killed
        # mid-request and taking every other endpoint down with us.
        raise HTTPException(503, (
            "Live re-analysis is disabled in this deployment. Results are "
            "precomputed at build time; a container this size cannot run the "
            "pipeline. See scripts/precompute.py."
        ))
    try:
        analysis = store.get(scene_id, force=True)
    except KeyError:
        raise HTTPException(404, f"Unknown scene: {scene_id}")
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")
    return JSONResponse(scene_collection(analysis))


def _find_candidate(store: AnalysisStore, candidate_id: str):
    """Locate a candidate and its attribution across all analysed scenes."""
    scene_id = candidate_id.rsplit("-", 1)[0]
    scene_ids = [scene_id] if scene_id in store.manifests else list(store.manifests)

    for sid in scene_ids:
        try:
            analysis = store.get(sid)
        except Exception:
            continue
        for cand in analysis.candidates:
            if cand.candidate_id == candidate_id:
                attribution = next(
                    (a for a in analysis.attributions if a.candidate_id == candidate_id),
                    None,
                )
                return cand, attribution, analysis
    return None, None, None


@app.get("/api/slicks/{candidate_id}")
def slick_detail(candidate_id: str):
    """Full detail for one slick: physics, drift origin, ranked vessels."""
    store = get_store()
    cand, attribution, analysis = _find_candidate(store, candidate_id)
    if cand is None:
        raise HTTPException(404, f"Unknown slick: {candidate_id}")
    if attribution is None:
        raise HTTPException(500, f"No attribution recorded for {candidate_id}")
    return JSONResponse(attribution_detail(
        cand, attribution,
        synthetic=bool(analysis.stats.get("synthetic")) if analysis else False))


@app.get("/api/slicks/{candidate_id}/backtrace")
def backtrace(candidate_id: str):
    """Frames for the backwards-drift animation, plus the vessel tracks.

    Each frame carries the timestamp it represents, so the UI can show the
    date and time as the oil crawls back to its origin.
    """
    store = get_store()
    cand, attribution, analysis = _find_candidate(store, candidate_id)
    if cand is None:
        raise HTTPException(404, f"Unknown slick: {candidate_id}")
    if attribution is None or attribution.origin is None:
        raise HTTPException(
            409,
            f"No drift origin for {candidate_id}"
            + (f" - {attribution.abstain_reason}" if attribution else ""),
        )

    origin = attribution.origin
    detail = attribution_detail(
        cand, attribution,
        synthetic=bool(analysis.stats.get("synthetic")) if analysis else False)
    return JSONResponse({
        "candidate_id": candidate_id,
        "observed_at": cand.wind.source and detail["evidence"].get("drift", {}).get("estimated_at"),
        "origin": detail["origin"],
        "frames": origin.track,          # oldest first
        "n_frames": len(origin.track),
        "backtrack_hours": origin.backtrack_hours,
        "uncertainty_km": origin.uncertainty_km,
        "method": origin.method,
        "reliable": origin.is_reliable,
        "vessels": detail["vessels"],
        "slick_polygon": detail["slick"]["polygon"],
        "disclaimer": DISCLAIMER,
        "caveat": (
            "Backward drift is advection only; weathering is irreversible and "
            "is disabled. Origin uncertainty grows with backtrack time."
        ),
    })


@app.get("/api/slicks/{candidate_id}/timeline")
def timeline(candidate_id: str, at: str = Query(None, description="ISO time; default now")):
    """The slick at three moments: where it started, what SAR saw, where it is now.

        origin  --(backward)--  observed  --(forward)--  now

    The present position is what a response vessel needs: imagery is 3-24 h
    old by the time we see it, so the slick has moved since acquisition.
    """
    from datetime import datetime, timezone

    store = get_store()
    cand, attribution, analysis = _find_candidate(store, candidate_id)
    if cand is None:
        raise HTTPException(404, f"Unknown slick: {candidate_id}")

    target = None
    if at:
        try:
            target = datetime.fromisoformat(at)
            target = target if target.tzinfo else target.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(400, f"Bad time: {at!r}")
    target = target or datetime.now(timezone.utc)

    observed_at = analysis.scene.acquired_at
    config = store._config_for(store.manifest(analysis.scene.scene_id) or {})

    from api.serialize import _wkt_to_coords
    from drift.forward import DriftState, forward_drift
    from drift.runner import _parse_polygon, build_fields

    states: list[dict] = []

    # 1. Origin - from the backward run already computed.
    origin = attribution.origin if attribution else None
    if origin is not None:
        states.append(DriftState(
            label="origin", lon=origin.lon, lat=origin.lat,
            at=origin.estimated_at, uncertainty_km=origin.uncertainty_km,
            hours_from_observation=-origin.backtrack_hours,
            description=(
                f"estimated release point, {origin.backtrack_hours:.0f} h before "
                f"acquisition, from backward drift"
            ),
        ).to_dict())

    # 2. Observed - what the satellite actually recorded. No modelling.
    states.append(DriftState(
        label="observed", lon=cand.centroid[0], lat=cand.centroid[1],
        at=observed_at, uncertainty_km=0.0, hours_from_observation=0.0,
        area_km2=cand.area_km2,
        description="position recorded by Sentinel-1 - an observation, not a model output",
        polygon=_wkt_to_coords(cand.polygon_wkt),
    ).to_dict())

    # 3. Now - advected forward from the observation.
    forward_warnings: list[str] = []
    forward_reliable = True
    try:
        # scene_id matters: a config using drift.currents_dir holds one field
        # per scene, and without the id there is nothing to look up. Omitting
        # it made every timeline fall back to "forward drift failed".
        currents, wind_field = build_fields(
            config, wind_speed_ms=cand.wind.speed_ms,
            wind_direction_deg=cand.wind.direction_deg,
            scene_id=cand.scene_id,
        )
        result = forward_drift(
            polygon_lonlat=_parse_polygon(cand.polygon_wkt),
            observed_at=observed_at,
            until=target,
            currents=currents,
            wind=wind_field,
            timestep_minutes=float(config.get("drift.timestep_minutes", 30.0)),
            n_particles=int(config.get("drift.n_particles", 200)),
            diffusion_m2_s=float(config.get("drift.diffusion_m2_s", 5.0)),
        )
        states.append(result.state.to_dict())
        forward_track = result.track
        forward_warnings = result.warnings
        forward_reliable = result.reliable
    except Exception as exc:
        # Never invent a present position; say the forecast failed.
        log.error("Forward drift failed for %s: %s", candidate_id, exc)
        forward_track = []
        forward_warnings = [f"forward drift unavailable: {exc}"]
        forward_reliable = False

    return JSONResponse({
        "candidate_id": candidate_id,
        "observed_at": observed_at.isoformat(),
        "target_time": target.isoformat(),
        "age_hours": round((target - observed_at).total_seconds() / 3600.0, 2),
        "states": states,
        "backward_track": (origin.track if origin else []),
        "forward_track": forward_track,
        "forward_reliable": forward_reliable,
        "warnings": forward_warnings,
        "vessels": [
            {
                "rank": i + 1, "mmsi": v.mmsi, "name": v.name,
                "went_dark": v.went_dark, "score": v.score, "track": v.track,
            }
            for i, v in enumerate(attribution.candidates if attribution else [])
        ],
        "caveat": (
            "The present position is advection only - currents and wind, no "
            "weathering. Uncertainty grows with time since acquisition."
        ),
    })


@app.get("/api/incidents")
def incidents(
    bbox: str = Query(None, description="min_lon,min_lat,max_lon,max_lat"),
    since: str = Query(None, description="ISO date; default 2014-10-01, the Sentinel-1 era"),
    until: str = Query(None, description="ISO date"),
    limit: int = Query(3000, ge=1, le=20000),
    petroleum_only: bool = Query(True),
):
    """Documented oil spills worldwide - CONFIRMED events, not detections.

    This is the layer that makes the map genuinely global. Every entry is a
    real, recorded incident from a public registry, so it answers "show me
    actual oil spills" without any model being involved.
    """
    store = get_store()
    registry = store.registry
    if registry is None:
        raise HTTPException(
            503,
            "Incident registry not loaded. Run: python scripts/fetch_incidents.py",
        )

    from datetime import datetime, timezone

    def parse(value: str | None, default=None):
        if not value:
            return default
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(400, f"Bad date: {value!r}")

    start = parse(since, datetime(2014, 10, 1, tzinfo=timezone.utc))
    end = parse(until)

    box = None
    if bbox:
        try:
            values = [float(v) for v in bbox.split(",")]
            if len(values) != 4:
                raise ValueError
            box = values
        except ValueError:
            raise HTTPException(400, "bbox must be min_lon,min_lat,max_lon,max_lat")

    features, oldest, newest = [], None, None
    for incident in registry.incidents:
        if petroleum_only and not incident.is_petroleum:
            continue
        if box and not (box[0] <= incident.lon <= box[2] and box[1] <= incident.lat <= box[3]):
            continue
        when = incident.occurred_at
        persistent = bool(incident.extras.get("persistent"))
        if when is not None and not persistent:
            if start and when < start:
                continue
            if end and when > end:
                continue
        if when is not None:
            oldest = when if oldest is None or when < oldest else oldest
            newest = when if newest is None or when > newest else newest

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [incident.lon, incident.lat]},
            "properties": {
                **incident.to_dict(),
                "persistent": persistent,
                "natural_seep": bool(incident.extras.get("natural_seep")),
                "kind": "documented_incident",
            },
        })
        if len(features) >= limit:
            break

    return JSONResponse({
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "count": len(features),
            "registry_size": len(registry),
            "date_range": [
                oldest.isoformat() if oldest else None,
                newest.isoformat() if newest else None,
            ],
            "sources": sorted({i.source for i in registry.incidents}),
            "note": (
                "These are CONFIRMED spill events from public incident "
                "registries, not model detections. Coverage is strongest in "
                "US waters (NOAA), supplemented by a curated world catalogue."
            ),
        },
    })


@app.get("/api/stats")
def stats():
    """Dashboard summary across every analysed scene and the incident registry."""
    store = get_store()
    from collections import Counter

    analysed, confirmed, rejected, abstained, attributed, dark = 0, 0, 0, 0, 0, 0
    reject_reasons: Counter[str] = Counter()
    stage_times: dict[str, float] = {}
    corroborated = 0

    for scene_id in store.manifests:
        try:
            analysis = store.get(scene_id)
        except Exception:
            continue
        analysed += 1
        confirmed += analysis.stats.get("n_confirmed", 0)
        rejected += analysis.stats.get("n_rejected", 0)
        for candidate in analysis.rejected:
            reason = (candidate.rejected_reason or "").lower()
            if "wind" in reason:
                reject_reasons["low or high wind"] += 1
            elif "internal-wave" in reason:
                reject_reasons["internal-wave train"] += 1
            elif "round" in reason or "blob" in reason:
                reject_reasons["blob shape (bloom)"] += 1
            elif "granular" in reason or "texture" in reason:
                reject_reasons["texture (rain cell)"] += 1
            elif "damping" in reason:
                reject_reasons["weak damping"] += 1
            else:
                reject_reasons["other physics"] += 1
        # Only slicks where naming a vessel was ever on the table. Counting
        # every attribution swept in the rejected look-alikes - which have no
        # vessel by definition - and drove the rate to 100% while 1 of 34
        # confirmed slicks was in fact attributed. Fixed sources are excluded
        # too: declining to blame a ship for a documented wellhead is the
        # correct answer, not an abstention.
        attributable = {c.candidate_id for c in analysis.confirmed}
        for attribution in analysis.attributions:
            # "unknown" counts: morphology could not route it, but a vessel
            # could still have been named, so an abstention there is a real
            # abstention. Only the fixed sources are out of scope.
            in_scope = (attribution.candidate_id in attributable
                        and attribution.source_type
                        not in ("natural_seep", "infrastructure"))
            if in_scope and attribution.abstained:
                abstained += 1
            elif in_scope and attribution.candidates:
                attributed += 1
            if any(c.went_dark for c in attribution.candidates):
                dark += 1
            if (attribution.evidence.get("corroboration") or {}).get("confirmed"):
                corroborated += 1
        for stage, seconds in analysis.timings.items():
            stage_times[stage] = round(stage_times.get(stage, 0.0) + seconds, 3)

    registry_size = len(store.registry) if store.registry else 0
    total_decisions = abstained + attributed

    return JSONResponse({
        "scenes_analysed": analysed,
        "slicks_confirmed": confirmed,
        "lookalikes_rejected": rejected,
        "rejection_reasons": dict(reject_reasons.most_common()),
        "attributed": attributed,
        "abstained": abstained,
        "abstention_rate": round(abstained / total_decisions, 3) if total_decisions else 0.0,
        "dark_vessel_flags": dark,
        "corroborated_by_registry": corroborated,
        "documented_incidents": registry_size,
        "stage_seconds": stage_times,
        "total_seconds": round(sum(stage_times.values()), 2),
    })


@app.get("/api/cerulean")
def cerulean(
    bbox: str = Query(None, description="min_lon,min_lat,max_lon,max_lat"),
    limit: int = Query(50, ge=1, le=500),
):
    """Slicks from SkyTruth Cerulean - an independent reference, not truth.

    CLAUDE.md rule 7: never train on these. They are probable sources, and
    are shown for comparison only.
    """
    import requests

    params: dict[str, Any] = {"limit": limit}
    if bbox:
        try:
            values = [float(v) for v in bbox.split(",")]
            if len(values) != 4:
                raise ValueError
        except ValueError:
            raise HTTPException(400, "bbox must be min_lon,min_lat,max_lon,max_lat")
        params["bbox"] = ",".join(str(v) for v in values)

    try:
        resp = requests.get(
            "https://api.cerulean.skytruth.org/collections/public.slick/items",
            params=params, timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(
            502, f"Cerulean unavailable: {exc}. This is an external reference; "
                 f"the pipeline does not depend on it."
        )

    data.setdefault("meta", {})["note"] = (
        "SkyTruth Cerulean detections, shown for comparison. These are probable "
        "sources, never used as training ground truth."
    )
    return JSONResponse(data)


# The UI is served from the same origin so a local demo is a single command.
# Set SERVE_UI=false when the frontend is deployed separately - then this
# process is a pure API and the two tiers scale independently.
SERVE_UI = os.environ.get("SERVE_UI", "true").lower() not in ("false", "0", "no")
UI_DIR = REPO_ROOT / "ui"
if SERVE_UI and UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

    @app.get("/")
    def index() -> FileResponse:
        page = UI_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "UI not built")
        return FileResponse(str(page))
