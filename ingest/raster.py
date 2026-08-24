"""Reading Sentinel-1 rasters without holding a whole GRD in memory.

A GRD scene is ~1 GB per polarisation. We read windowed and decimated
straight off disk via rasterio, so the full-resolution array never exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RasterData:
    array: np.ndarray
    bbox: tuple[float, float, float, float] | None
    crs: str | None
    nodata: float | None
    path: Path
    native_shape: tuple[int, int]
    decimation: int = 1


def _bounds_to_bbox(bounds, crs) -> tuple[float, float, float, float] | None:
    """Reproject raster bounds to EPSG:4326 lon/lat."""
    try:
        from rasterio.warp import transform_bounds

        if crs is None:
            return None
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326", *bounds, densify_pts=21
        )
        return (float(west), float(south), float(east), float(north))
    except Exception as exc:  # pragma: no cover - depends on CRS metadata
        log.warning("Could not reproject bounds to EPSG:4326: %s", exc)
        return None


def read_raster(path: str | Path, decimation: int = 1, band: int = 1) -> RasterData:
    """Read a raster, optionally decimated by an integer factor on read.

    Decimating during the read is what keeps a 1 GB GRD off the heap: rasterio
    pulls only the requested overview-sized buffer from disk.
    """
    import rasterio

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raster not found: {p}")

    with rasterio.open(p) as src:
        native = (src.height, src.width)
        if decimation > 1:
            out_shape = (
                max(1, src.height // decimation),
                max(1, src.width // decimation),
            )
            arr = src.read(band, out_shape=out_shape, resampling=_avg_resampling())
        else:
            arr = src.read(band)
        bbox = _bounds_to_bbox(src.bounds, src.crs)
        crs = str(src.crs) if src.crs else None
        nodata = src.nodata

    return RasterData(
        array=np.asarray(arr, dtype=np.float32),
        bbox=bbox,
        crs=crs,
        nodata=nodata,
        path=p,
        native_shape=native,
        decimation=decimation,
    )


def _avg_resampling():
    """Averaging resampler - nearest-neighbour decimation would keep speckle."""
    from rasterio.enums import Resampling

    return Resampling.average


def write_raster(
    path: str | Path,
    array: np.ndarray,
    bbox: tuple[float, float, float, float] | None = None,
    dtype: str = "float32",
) -> Path:
    """Write a north-up EPSG:4326 GeoTIFF. Used for cached intermediates."""
    import rasterio
    from rasterio.transform import from_bounds

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=dtype)
    if arr.ndim == 2:
        arr = arr[None, ...]
    count, height, width = arr.shape

    transform = (
        from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], width, height)
        if bbox
        else from_bounds(0, 0, width, height, width, height)
    )

    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs="EPSG:4326" if bbox else None,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(arr)
    return p


def find_safe_measurement(safe_dir: Path, polarisation: str) -> Path | None:
    """Locate the measurement GeoTIFF for one polarisation inside a .SAFE."""
    measurement = safe_dir / "measurement"
    if not measurement.is_dir():
        return None
    hits = sorted(measurement.glob(f"*-{polarisation.lower()}-*.tiff"))
    return hits[0] if hits else None
