"""The ingest stage, end to end, with caching.

    GRD -> Sigma0 dB -> speckle filter (heavy + light) -> land mask -> tiles

Caching is aggressive on purpose: re-tiling a scene is slow and re-running it
by accident burns hours. The cache key covers every parameter that changes
the output, so a config edit invalidates it automatically rather than
silently serving stale tiles.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.config import Config, resolve_path
from core.contracts import Scene
from ingest.calibrate import to_sigma0_db
from ingest.landmask import build_land_mask
from ingest.raster import find_safe_measurement, read_raster
from ingest.speckle import filter_pair, speckle_index
from ingest.tiling import TileGrid, build_tiles, downsample_factor_for

log = logging.getLogger(__name__)

CACHE_VERSION = "v1"


@dataclass
class IngestResult:
    """Everything downstream stages need from a scene."""

    scene: Scene
    grid: TileGrid
    sigma0_db: dict[str, np.ndarray]   # heavily filtered, downsampled
    light_db: dict[str, np.ndarray]    # lightly filtered - for texture only
    land_mask: np.ndarray              # downsampled
    bbox: tuple[float, float, float, float]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def is_dual_pol(self) -> bool:
        return "vh" in self.sigma0_db


def cache_key(scene: Scene, params: dict[str, Any]) -> str:
    """Stable hash over scene identity and every output-affecting parameter."""
    payload = {
        "version": CACHE_VERSION,
        "scene_id": scene.scene_id,
        "vv": str(scene.vv_path),
        "vh": str(scene.vh_path) if scene.vh_path else None,
        "params": params,
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_channel(path: Path, decimation: int, polarisation: str):
    """Read one polarisation and calibrate it to Sigma0 dB."""
    safe_dir = None
    p = Path(path)
    # Walk up looking for the .SAFE root so the calibration LUT can be found.
    for parent in [p.parent, *p.parents]:
        if parent.name.endswith(".SAFE"):
            safe_dir = parent
            break

    raster = read_raster(p, decimation=decimation)
    cal = to_sigma0_db(
        raster.array, safe_dir=safe_dir, polarisation=polarisation, nodata=raster.nodata
    )
    return cal.sigma0_db, raster


def _native_resolution_m(scene: Scene) -> float:
    """Ground pixel size in metres, from the scene bbox and raster shape.

    Falls back to the Sentinel-1 GRD IW nominal 10 m only when the raster
    carries no usable geolocation.
    """
    import math

    from ingest.tiling import S1_IW_GRD_PIXEL_M

    try:
        vv = Path(scene.vv_path)
        if vv.is_dir() and vv.name.endswith(".SAFE"):
            found = find_safe_measurement(vv, "vv")
            if found is None:
                return S1_IW_GRD_PIXEL_M
            vv = found
        import rasterio

        with rasterio.open(vv) as src:
            height, width = src.height, src.width
            bounds, crs = src.bounds, src.crs
    except Exception:
        return S1_IW_GRD_PIXEL_M

    if crs is not None and not crs.is_geographic:
        # Projected: the transform already carries metres.
        return float(abs(bounds.right - bounds.left) / max(width, 1))

    bbox = scene.bbox
    if crs is not None:
        from ingest.raster import _bounds_to_bbox

        bbox = _bounds_to_bbox(bounds, crs) or scene.bbox
    if bbox is None:
        return S1_IW_GRD_PIXEL_M

    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    x_m = 111_320.0 * math.cos(math.radians(mid_lat)) * (max_lon - min_lon) / max(width, 1)
    y_m = 110_574.0 * (max_lat - min_lat) / max(height, 1)
    res = (abs(x_m) + abs(y_m)) / 2.0
    return res if res > 0.1 else S1_IW_GRD_PIXEL_M


def ingest_scene(scene: Scene, config: Config, use_cache: bool = True) -> IngestResult:
    """Run the full ingest stage for one scene."""
    ing = config.section("ingest")
    tile_size = int(ing.get("tile_size", 512))
    overlap = float(ing.get("tile_overlap", 0.25))
    target_res = float(ing.get("target_resolution_m", 80.0))
    method = str(ing.get("speckle_filter", "refined_lee"))
    window = int(ing.get("speckle_window", 7))

    params = {
        "tile_size": tile_size,
        "overlap": overlap,
        "target_res": target_res,
        "speckle": method,
        "window": window,
    }
    key = cache_key(scene, params)
    cache_dir = resolve_path(ing.get("cache_dir", "data/cache"))
    cache_path = cache_dir / f"{scene.scene_id}_{key}.pkl"

    if use_cache and cache_path.exists():
        log.info("Ingest cache HIT %s", cache_path.name)
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    started = time.perf_counter()

    # The downsample factor is derived from the raster's ACTUAL pixel size,
    # never assumed. A Sentinel-1 GRD IW is ~10 m, but cropped exports,
    # already-multi-looked products and test scenes are not, and decimating
    # a 50 m product by another 8x erases the very slicks we are looking for.
    native_res = _native_resolution_m(scene)
    total_factor = downsample_factor_for(
        source_res_m=native_res, target_res_m=target_res
    )
    read_decimation = max(1, total_factor // 2)
    residual = max(1, total_factor // read_decimation)
    log.info(
        "Native resolution ~%.0f m/px -> downsample %dx (read %dx, block %dx) "
        "for a %.0f m target",
        native_res, total_factor, read_decimation, residual, target_res,
    )

    vv_path = Path(scene.vv_path)
    if vv_path.is_dir() and vv_path.name.endswith(".SAFE"):
        found = find_safe_measurement(vv_path, "vv")
        if found is None:
            raise FileNotFoundError(f"No VV measurement inside {vv_path}")
        vv_path = found

    vv_db, vv_raster = _load_channel(vv_path, read_decimation, "vv")
    channels_raw: dict[str, np.ndarray] = {"vv": vv_db}

    if scene.vh_path is not None:
        vh_path = Path(scene.vh_path)
        if vh_path.is_dir() and vh_path.name.endswith(".SAFE"):
            found = find_safe_measurement(vh_path, "vh")
            if found is None:
                raise FileNotFoundError(f"No VH measurement inside {vh_path}")
            vh_path = found
        vh_db, _ = _load_channel(vh_path, read_decimation, "vh")
        if vh_db.shape != vv_db.shape:
            raise ValueError(
                f"VV shape {vv_db.shape} != VH shape {vh_db.shape} for {scene.scene_id}"
            )
        channels_raw["vh"] = vh_db
    else:
        log.info("Scene %s is single-pol; VH features degrade", scene.scene_id)

    bbox = vv_raster.bbox or scene.bbox
    if bbox is None:
        raise ValueError(f"Scene {scene.scene_id} has no geolocation")

    heavy: dict[str, np.ndarray] = {}
    light: dict[str, np.ndarray] = {}
    for name, db in channels_raw.items():
        h, lt = filter_pair(db, method=method, size=window)
        heavy[name] = h
        light[name] = lt

    land = build_land_mask(heavy["vv"], bbox=bbox)

    grid = build_tiles(
        channels=heavy,
        land_mask=land,
        tile_size=tile_size,
        overlap=overlap,
        downsample=residual,
        # The array is already decimated by read_decimation, so the tile grid
        # must be told the post-read pixel size or resolution_m comes out
        # wrong by that factor - and every area in km2 downstream with it.
        source_res_m=native_res * read_decimation,
        bbox=bbox,
    )

    # Keep scene-sized downsampled copies for the stages that work on
    # polygons rather than tiles (look-alike features, damping ratio).
    from ingest.tiling import downsample_db, downsample_mask

    small_heavy = {k: downsample_db(v, residual) for k, v in heavy.items()}
    small_light = {k: downsample_db(v, residual) for k, v in light.items()}
    small_land = downsample_mask(land, residual)

    elapsed = time.perf_counter() - started
    stats = {
        "elapsed_s": round(elapsed, 2),
        "n_tiles": len(grid),
        "resolution_m": grid.resolution_m,
        "downsample_total": read_decimation * residual,
        "land_fraction": round(float(small_land.mean()), 4),
        "speckle_index_before": round(speckle_index(channels_raw["vv"]), 4),
        "speckle_index_after": round(speckle_index(heavy["vv"]), 4),
        "dual_pol": "vh" in heavy,
        "calibration_bbox": bbox,
    }
    log.info(
        "Ingested %s in %.2fs -> %d tiles at %.0f m/px",
        scene.scene_id, elapsed, len(grid), grid.resolution_m,
    )

    result = IngestResult(
        scene=scene,
        grid=grid,
        sigma0_db=small_heavy,
        light_db=small_light,
        land_mask=small_land,
        bbox=bbox,
        stats=stats,
    )

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)  # atomic: a killed run leaves no half cache
        log.info("Ingest cached -> %s", cache_path.name)

    return result


def ingest_arrays(
    scene: Scene,
    vv_db: np.ndarray,
    vh_db: np.ndarray | None,
    config: Config,
) -> IngestResult:
    """Ingest from in-memory dB arrays, skipping raster IO.

    Used by the synthetic demo generator and by tests, so the whole pipeline
    can be exercised without a 1 GB download.
    """
    ing = config.section("ingest")
    method = str(ing.get("speckle_filter", "refined_lee"))
    window = int(ing.get("speckle_window", 7))
    tile_size = int(ing.get("tile_size", 512))
    overlap = float(ing.get("tile_overlap", 0.25))

    channels = {"vv": np.asarray(vv_db, dtype=np.float32)}
    if vh_db is not None:
        channels["vh"] = np.asarray(vh_db, dtype=np.float32)

    heavy: dict[str, np.ndarray] = {}
    light: dict[str, np.ndarray] = {}
    for name, db in channels.items():
        h, lt = filter_pair(db, method=method, size=window)
        heavy[name] = h
        light[name] = lt

    land = build_land_mask(heavy["vv"], bbox=scene.bbox)
    grid = build_tiles(
        channels=heavy,
        land_mask=land,
        tile_size=tile_size,
        overlap=overlap,
        downsample=1,
        bbox=scene.bbox,
    )
    return IngestResult(
        scene=scene,
        grid=grid,
        sigma0_db=heavy,
        light_db=light,
        land_mask=land,
        bbox=scene.bbox,
        stats={
            "n_tiles": len(grid),
            "resolution_m": grid.resolution_m,
            "dual_pol": vh_db is not None,
            "source": "in-memory arrays",
        },
    )
