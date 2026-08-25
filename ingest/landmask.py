"""Land masking.

Land is bright and structured on SAR; leaving it in produces a flood of
false "dark patch" candidates along every shadowed valley and inland water
body. We mask it out before anything else looks at the scene.

Primary source is a bundled global coastline grid (`global_land_mask`,
derived from GSHHG), which works offline and needs no download. Where a
GSHHG shapefile is available (OpenDrift bundles one) it can be supplied
explicitly for higher-resolution coastlines.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# global_land_mask allocates its whole coastline grid on import: a
# 21600 x 43200 boolean array, 933 MB resident. That is affordable inside a
# training or analysis run and ruinous anywhere else - a 512 MB API container
# is killed by the import alone, before it serves anything. So it is loaded on
# first use and cached, never at import time.
_globe = None
_GLOBE_TRIED = False


def _load_globe():
    """The bundled coastline grid, loaded once on first use.

    Returns None when the package is missing; callers fall back to radiometric
    land detection.
    """
    global _globe, _GLOBE_TRIED
    if _GLOBE_TRIED:
        return _globe
    _GLOBE_TRIED = True
    try:
        from global_land_mask import globe

        _globe = globe
    except ImportError:  # pragma: no cover - exercised only on a broken install
        log.warning("global_land_mask unavailable; "
                    "falling back to radiometric land detection")
    return _globe

# Buffer applied around detected land. Coastal returns bleed several pixels
# offshore, and a slick "detected" in that bleed is always spurious.
DEFAULT_COAST_BUFFER_PX = 3


def geographic_land_mask(
    lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """True where the coordinate falls on land, from the bundled coastline."""
    globe = _load_globe()
    if globe is None:
        return np.zeros(np.shape(lons), dtype=bool)
    lat = np.clip(np.asarray(lats, dtype=np.float64), -90.0, 90.0)
    lon = np.asarray(lons, dtype=np.float64)
    lon = (lon + 180.0) % 360.0 - 180.0
    return np.asarray(globe.is_land(lat, lon), dtype=bool)


def land_mask_for_bbox(
    bbox: tuple[float, float, float, float], shape: tuple[int, int]
) -> np.ndarray:
    """Rasterise the coastline over a bbox at the given array shape.

    bbox is (min_lon, min_lat, max_lon, max_lat); row 0 is the NORTH edge,
    matching the north-up convention of every GeoTIFF we read.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    height, width = shape
    lons = np.linspace(min_lon, max_lon, width)
    lats = np.linspace(max_lat, min_lat, height)  # north-up
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return geographic_land_mask(lon_grid, lat_grid)


def radiometric_land_mask(
    sigma0_db: np.ndarray,
    bright_threshold_db: float = -5.0,
    texture_window: int = 15,
    texture_threshold: float = 2.5,
) -> np.ndarray:
    """Fallback land detection from the imagery alone.

    Land is both brighter and far more texturally varied than open sea. Used
    only where no geolocation is available, and it is a weak substitute: it
    will also flag very rough seas and dense ship clusters.
    """
    from scipy import ndimage

    arr = np.asarray(sigma0_db, dtype=np.float32)
    local_std = ndimage.generic_filter(
        arr, np.std, size=texture_window, mode="reflect"
    ) if arr.size <= 512 * 512 else _fast_local_std(arr, texture_window)
    bright = arr > bright_threshold_db
    textured = local_std > texture_threshold
    return bright & textured


def _fast_local_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Uniform-filter local std — generic_filter is far too slow on a full scene."""
    from scipy import ndimage

    a = arr.astype(np.float64)
    mean = ndimage.uniform_filter(a, size=size, mode="reflect")
    mean_sq = ndimage.uniform_filter(a * a, size=size, mode="reflect")
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def buffer_mask(mask: np.ndarray, pixels: int = DEFAULT_COAST_BUFFER_PX) -> np.ndarray:
    """Dilate a boolean mask by `pixels` to swallow coastal bleed."""
    if pixels <= 0:
        return mask
    from scipy import ndimage

    return ndimage.binary_dilation(mask, iterations=int(pixels))


def build_land_mask(
    sigma0_db: np.ndarray,
    bbox: tuple[float, float, float, float] | None = None,
    buffer_px: int = DEFAULT_COAST_BUFFER_PX,
) -> np.ndarray:
    """Best available land mask for a scene, buffered.

    Uses the coastline where the scene is geolocated, and falls back to
    radiometry only when it is not.
    """
    # _load_globe() pays the grid's load cost, which is correct here: this
    # branch is about to rasterise the coastline anyway.
    if bbox is not None and _load_globe() is not None:
        mask = land_mask_for_bbox(bbox, sigma0_db.shape)
        source = "coastline"
    else:
        mask = radiometric_land_mask(sigma0_db)
        source = "radiometric-fallback"

    mask = buffer_mask(mask, buffer_px)
    log.info(
        "Land mask (%s): %.1f%% of scene masked", source, 100.0 * float(mask.mean())
    )
    return mask
