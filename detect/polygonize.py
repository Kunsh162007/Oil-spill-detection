"""Turn a probability mask into slick polygons carrying physical features.

Everything the look-alike stage reasons about is measured here, on the
scene-sized downsampled arrays rather than per tile, so a slick that spanned
several tiles is measured once as one object.

Texture features are deliberately taken from the LIGHTLY filtered image.
GLCM statistics computed on the aggressively filtered copy largely measure
the filter, not the sea. See ingest/speckle.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ingest.calibrate import db_to_linear
from ingest.tiling import pixel_to_lonlat

log = logging.getLogger(__name__)

# Width of the sea annulus sampled around a slick for the damping ratio,
# in downsampled pixels (~80 m each), so 8 px is roughly 640 m of context.
SURROUND_DILATION_PX = 8


@dataclass
class RegionFeatures:
    """Measured properties of one connected dark region."""

    label: int
    pixel_count: int
    area_km2: float
    centroid_rc: tuple[float, float]
    centroid_lonlat: tuple[float, float]
    elongation: float
    compactness: float
    orientation_deg: float      # long-axis bearing, degrees from north
    major_axis_km: float
    minor_axis_km: float
    damping_ratio: float
    mean_db: float
    surround_db: float
    texture_homogeneity: float
    texture_contrast: float
    texture_variance: float
    vh_vv_ratio: float | None
    mean_confidence: float
    polygon_lonlat: list[tuple[float, float]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


def _glcm_features(patch: np.ndarray, levels: int = 32) -> tuple[float, float, float]:
    """GLCM homogeneity, contrast and variance for one region patch.

    Quantised to 32 grey levels over a fixed dB range so the numbers are
    comparable between regions and between scenes. Per-patch autoscaling
    would make a uniformly dark slick look identical to uniform open sea.
    """
    from skimage.feature import graycomatrix, graycoprops

    if patch.size < 9:
        return 0.0, 0.0, 0.0

    lo, hi = -35.0, 0.0
    q = np.clip((patch - lo) / (hi - lo), 0.0, 1.0)
    q = (q * (levels - 1)).astype(np.uint8)

    # Four directions, averaged: slicks have no canonical orientation and a
    # single-angle GLCM would just measure which way the streak happens to lie.
    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    variance = float(np.var(patch))
    return homogeneity, contrast, variance


def _surrounding_stats(
    linear: np.ndarray,
    region_mask: np.ndarray,
    all_dark: np.ndarray,
    sea_mask: np.ndarray,
    dilation: int = SURROUND_DILATION_PX,
) -> float:
    """Mean linear backscatter of the sea immediately around a region.

    Other dark regions are excluded from the annulus. Without that, two
    slicks lying side by side each measure the other as "background sea" and
    both damping ratios collapse towards 1.0.
    """
    from scipy import ndimage

    grown = ndimage.binary_dilation(region_mask, iterations=dilation)
    annulus = grown & ~region_mask & sea_mask & ~all_dark
    if not annulus.any():
        annulus = grown & ~region_mask & sea_mask
    if not annulus.any():
        return float("nan")
    return float(np.mean(linear[annulus]))


def _contour_lonlat(
    region_mask: np.ndarray,
    shape: tuple[int, int],
    bbox: tuple[float, float, float, float],
    simplify_px: float = 1.5,
) -> list[tuple[float, float]]:
    """Trace a region outline and convert it to lon/lat vertices."""
    from skimage import measure

    padded = np.pad(region_mask.astype(np.uint8), 1, constant_values=0)
    contours = measure.find_contours(padded, 0.5)
    if not contours:
        return []
    contour = max(contours, key=len) - 1.0  # undo the pad offset

    if simplify_px > 0 and len(contour) > 8:
        contour = measure.approximate_polygon(contour, tolerance=simplify_px)

    pts = [pixel_to_lonlat(float(r), float(c), shape, bbox) for r, c in contour]
    if len(pts) >= 3 and pts[0] != pts[-1]:
        pts.append(pts[0])  # close the ring, as GeoJSON/WKT require
    return pts


def polygon_wkt(points: list[tuple[float, float]]) -> str:
    """WKT POLYGON from lon/lat vertices. Empty ring -> POLYGON EMPTY."""
    if len(points) < 4:
        return "POLYGON EMPTY"
    ring = ", ".join(f"{lon:.6f} {lat:.6f}" for lon, lat in points)
    return f"POLYGON (({ring}))"


def _pixel_area_km2(resolution_m: float) -> float:
    return (resolution_m / 1000.0) ** 2


def _orientation_to_bearing(orientation_rad: float) -> float:
    """skimage orientation (rad, row/col frame) -> compass bearing 0-180.

    Calibrated empirically against known geometries (row 0 = north edge,
    col 0 = west edge):
        N-S streak   raw   0 deg -> bearing   0
        E-W streak   raw  90 deg -> bearing  90
        NE-SW streak raw -45 deg -> bearing  45
        NW-SE streak raw  45 deg -> bearing 135
    A slick axis is undirected, so the result folds into 0-180. Getting this
    wrong by 90 degrees would silently invert every parity score in
    attribute/scoring.py, so it is pinned by a test.
    """
    deg = math.degrees(orientation_rad)
    return (-deg) % 180.0


def extract_regions(
    probability: np.ndarray,
    sigma0_db: dict[str, np.ndarray],
    light_db: dict[str, np.ndarray],
    land_mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    resolution_m: float,
    threshold: float = 0.5,
    min_area_km2: float = 0.05,
) -> list[RegionFeatures]:
    """Threshold, label, and measure every dark region in a scene.

    Returns one RegionFeatures per connected component above the area floor,
    largest first.
    """
    from scipy import ndimage
    from skimage import measure

    if probability.shape != land_mask.shape:
        raise ValueError(
            f"probability {probability.shape} != land_mask {land_mask.shape}"
        )

    sea = ~land_mask
    dark = (probability >= threshold) & sea
    if not dark.any():
        log.info("No pixels above threshold %.2f", threshold)
        return []

    # Close single-pixel gaps so speckle does not shatter one slick into many.
    dark = ndimage.binary_closing(dark, structure=np.ones((3, 3)), iterations=1)
    dark &= sea

    labels, n_labels = ndimage.label(dark, structure=np.ones((3, 3)))
    if n_labels == 0:
        return []

    vv_db = sigma0_db["vv"]
    vv_linear = db_to_linear(vv_db)
    vv_light = light_db.get("vv", vv_db)
    vh_db = sigma0_db.get("vh")
    px_area = _pixel_area_km2(resolution_m)
    min_pixels = max(4, int(round(min_area_km2 / px_area)))

    props = {p.label: p for p in measure.regionprops(labels)}
    out: list[RegionFeatures] = []

    for label_id, prop in props.items():
        if prop.area < min_pixels:
            continue
        region_mask = labels == label_id
        feats = _measure_region(
            label_id, prop, region_mask, dark, sea, probability,
            vv_db, vv_linear, vv_light, vh_db, bbox, resolution_m, px_area,
        )
        out.append(feats)

    out.sort(key=lambda r: r.area_km2, reverse=True)
    log.info(
        "Extracted %d regions above %.3f km2 (from %d labelled)",
        len(out), min_area_km2, n_labels,
    )
    return out


def _measure_region(
    label_id, prop, region_mask, all_dark, sea, probability,
    vv_db, vv_linear, vv_light, vh_db, bbox, resolution_m, px_area,
) -> RegionFeatures:
    """Measure every physical feature for one region."""
    shape = vv_db.shape
    minr, minc, maxr, maxc = prop.bbox

    major = float(prop.axis_major_length)
    minor = float(prop.axis_minor_length)
    # A one-pixel-wide region has minor axis 0; clamp so elongation stays finite.
    elongation = major / max(minor, 1.0)

    perimeter = float(prop.perimeter) or 1.0
    compactness = 4.0 * math.pi * float(prop.area) / (perimeter**2)

    mean_linear = float(np.mean(vv_linear[region_mask]))
    surround_linear = _surrounding_stats(vv_linear, region_mask, all_dark, sea)
    damping = (
        float(mean_linear / surround_linear)
        if surround_linear and math.isfinite(surround_linear) and surround_linear > 0
        else float("nan")
    )

    patch = vv_light[minr:maxr, minc:maxc]
    homog, contrast, variance = _glcm_features(patch)

    vh_vv = None
    if vh_db is not None:
        vh_lin = float(np.mean(db_to_linear(vh_db)[region_mask]))
        if mean_linear > 0:
            vh_vv = float(vh_lin / mean_linear)

    cr, cc = float(prop.centroid[0]), float(prop.centroid[1])
    centroid_lonlat = pixel_to_lonlat(cr, cc, shape, bbox)
    poly = _contour_lonlat(region_mask, shape, bbox)

    km_per_px = resolution_m / 1000.0
    return RegionFeatures(
        label=int(label_id),
        pixel_count=int(prop.area),
        area_km2=float(prop.area) * px_area,
        centroid_rc=(cr, cc),
        centroid_lonlat=centroid_lonlat,
        elongation=float(elongation),
        compactness=float(min(compactness, 1.0)),
        orientation_deg=_orientation_to_bearing(float(prop.orientation)),
        major_axis_km=major * km_per_px,
        minor_axis_km=minor * km_per_px,
        damping_ratio=damping,
        mean_db=float(np.mean(vv_db[region_mask])),
        surround_db=(
            10.0 * math.log10(surround_linear)
            if surround_linear and math.isfinite(surround_linear) and surround_linear > 0
            else float("nan")
        ),
        texture_homogeneity=homog,
        texture_contrast=contrast,
        texture_variance=variance,
        vh_vv_ratio=vh_vv,
        mean_confidence=float(np.mean(probability[region_mask])),
        polygon_lonlat=poly,
    )
