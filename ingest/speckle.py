"""Speckle filtering for SAR.

Radar images are grainy by nature (coherent speckle, multiplicative). We
produce TWO outputs from every scene:

  * `filtered` — refined Lee, aggressive. Feeds the segmentation network.
  * `light`    — a mild 3x3 Lee. Feeds the texture features in
                 detect/lookalike.

That split is not tidiness. GLCM homogeneity/contrast measured on an
aggressively filtered image are largely a measurement of the filter, not of
the sea, and the look-alike stage is the one place where texture has to carry
real information.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

# Sentinel-1 GRD IW is multi-looked; equivalent number of looks ~4.4.
DEFAULT_ENL = 4.4


def _local_stats(img: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and variance via uniform filters."""
    mean = ndimage.uniform_filter(img, size=size, mode="reflect")
    mean_sq = ndimage.uniform_filter(img * img, size=size, mode="reflect")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, var


def lee_filter(img: np.ndarray, size: int = 7, enl: float = DEFAULT_ENL) -> np.ndarray:
    """Classic Lee adaptive filter.

    Blends towards the local mean in homogeneous areas and towards the
    original pixel where local variance exceeds what speckle alone explains,
    which is what preserves slick edges.
    """
    # float32 throughout: refined Lee runs eight directional convolutions, and
# in float64 those intermediates alone exceed the memory of a small
# container. Precision is far beyond what dB backscatter needs.
    arr = np.asarray(img, dtype=np.float32)
    mean, var = _local_stats(arr, size)

    # Speckle-only variance for this ENL, in intensity terms.
    cu2 = 1.0 / enl
    noise_var = cu2 * mean * mean

    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(var > 0, (var - noise_var) / var, 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    return (mean + weight * (arr - mean)).astype(np.float32)


# Eight directional edge templates (Lee 1981 / Lopes 1990 refined variant).
# Each mask selects the half-window on one side of an edge at that orientation.
def _directional_masks(size: int) -> list[np.ndarray]:
    r = size // 2
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    masks = []
    for angle_deg in range(0, 180, 45):
        theta = np.radians(angle_deg)
        # Signed distance from the line through the centre at this angle.
        d = xx * np.sin(theta) - yy * np.cos(theta)
        masks.append((d >= 0).astype(np.float64))
        masks.append((d <= 0).astype(np.float64))
    return masks


def refined_lee(img: np.ndarray, size: int = 7, enl: float = DEFAULT_ENL) -> np.ndarray:
    """Refined Lee: Lee statistics computed over the edge-aligned half-window.

    In a homogeneous patch this behaves like plain Lee. Near a slick boundary
    it picks whichever half-window is most homogeneous, so it smooths *along*
    the edge instead of across it — which matters because the thin tapering
    streak of a bilge dump is exactly the structure a symmetric filter erodes.
    """
    # float32 throughout: refined Lee runs eight directional convolutions, and
# in float64 those intermediates alone exceed the memory of a small
# container. Precision is far beyond what dB backscatter needs.
    arr = np.asarray(img, dtype=np.float32)
    if size < 3:
        return arr.astype(np.float32)

    masks = _directional_masks(size)
    cu2 = 1.0 / enl

    # Coefficient of variation over the full window decides whether the pixel
    # sits in a homogeneous region (use plain Lee) or on an edge.
    full_mean, full_var = _local_stats(arr, size)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(full_mean > 0, np.sqrt(full_var) / full_mean, 0.0)
    cv = np.nan_to_num(cv)
    cu = np.sqrt(cu2)
    # Above this the window is heterogeneous enough to warrant edge handling.
    cmax = np.sqrt(1.0 + 2.0 * cu2)
    edge_region = cv > cu

    best_mean = full_mean.copy()
    best_var = full_var.copy()
    best_cv = np.full(arr.shape, np.inf)

    for mask in masks:
        norm = mask.sum()
        if norm == 0:
            continue
        kernel = mask / norm
        m = ndimage.convolve(arr, kernel, mode="reflect")
        m2 = ndimage.convolve(arr * arr, kernel, mode="reflect")
        v = np.maximum(m2 - m * m, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(m > 0, np.sqrt(v) / m, np.inf)
        c = np.nan_to_num(c, nan=np.inf, posinf=np.inf)

        take = c < best_cv
        best_cv = np.where(take, c, best_cv)
        best_mean = np.where(take, m, best_mean)
        best_var = np.where(take, v, best_var)

    # Homogeneous pixels use the full window; edge pixels use the best half.
    mean = np.where(edge_region, best_mean, full_mean)
    var = np.where(edge_region, best_var, full_var)

    noise_var = cu2 * mean * mean
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(var > 0, (var - noise_var) / var, 0.0)
    weight = np.clip(np.nan_to_num(weight), 0.0, 1.0)

    out = mean + weight * (arr - mean)

    # Very bright isolated returns (ships, platforms) must survive untouched —
    # they are targets in their own right, not speckle.
    point_targets = cv > cmax
    out = np.where(point_targets, arr, out)

    return out.astype(np.float32)


def apply_speckle_filter(
    img: np.ndarray, method: str = "refined_lee", size: int = 7, enl: float = DEFAULT_ENL
) -> np.ndarray:
    if method == "none":
        return np.asarray(img, dtype=np.float32)
    if method == "lee":
        return lee_filter(img, size=size, enl=enl)
    if method == "refined_lee":
        return refined_lee(img, size=size, enl=enl)
    if method == "median":
        return ndimage.median_filter(np.asarray(img, dtype=np.float32), size=size)
    raise ValueError(f"Unknown speckle filter: {method!r}")


def filter_pair(
    img: np.ndarray, method: str = "refined_lee", size: int = 7, enl: float = DEFAULT_ENL
) -> tuple[np.ndarray, np.ndarray]:
    """Return (heavily_filtered, lightly_filtered).

    Callers must feed the light copy to texture features. See module docstring.
    """
    heavy = apply_speckle_filter(img, method=method, size=size, enl=enl)
    light = lee_filter(img, size=3, enl=enl)
    return heavy, light


def speckle_index(img: np.ndarray) -> float:
    """Coefficient of variation over the whole image.

    A crude but useful check that filtering did something: this number should
    drop noticeably after filtering and is ~1/sqrt(ENL) for pure speckle.
    """
    # float32 throughout: refined Lee runs eight directional convolutions, and
# in float64 those intermediates alone exceed the memory of a small
# container. Precision is far beyond what dB backscatter needs.
    arr = np.asarray(img, dtype=np.float32)
    m = float(arr.mean())
    return float(arr.std() / m) if m != 0 else 0.0
