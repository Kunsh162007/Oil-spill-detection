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

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from core.config import REPO_ROOT, load_config
from api.serialize import attribution_detail, scene_collection, world_index
from api.store import AnalysisStore
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
    config = load_config(CONFIG_PATH)
    store = AnalysisStore(config)
    manifests = store.discover()
    _state["store"] = store
    _state["config"] = config
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # local demo only
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    torch_cuda = False
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
    except ImportError:
        pass

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
        "disclaimer": DISCLAIMER,
        "timeliness": "near-real-time (imagery 3-24 h; free AIS ~72 h)",
    }


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


@app.get("/api/slicks")
def all_slicks(analyse: bool = Query(True, description="analyse any scene not yet processed")):
    """World index: confirmed slicks across every scene. Feeds the main map."""
    store = get_store()
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
    cand, attribution, _ = _find_candidate(store, candidate_id)
    if cand is None:
        raise HTTPException(404, f"Unknown slick: {candidate_id}")
    if attribution is None:
        raise HTTPException(500, f"No attribution recorded for {candidate_id}")
    return JSONResponse(attribution_detail(cand, attribution))


@app.get("/api/slicks/{candidate_id}/backtrace")
def backtrace(candidate_id: str):
    """Frames for the backwards-drift animation, plus the vessel tracks.

    Each frame carries the timestamp it represents, so the UI can show the
    date and time as the oil crawls back to its origin.
    """
    store = get_store()
    cand, attribution, _ = _find_candidate(store, candidate_id)
    if cand is None:
        raise HTTPException(404, f"Unknown slick: {candidate_id}")
    if attribution is None or attribution.origin is None:
        raise HTTPException(
            409,
            f"No drift origin for {candidate_id}"
            + (f" - {attribution.abstain_reason}" if attribution else ""),
        )

    origin = attribution.origin
    detail = attribution_detail(cand, attribution)
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


# The UI is served from the same origin so the demo is a single command.
UI_DIR = REPO_ROOT / "ui"
if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

    @app.get("/")
    def index() -> FileResponse:
        page = UI_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "UI not built")
        return FileResponse(str(page))
