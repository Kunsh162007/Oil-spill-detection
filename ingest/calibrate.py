"""Sentinel-1 GRD -> calibrated Sigma0 in dB.

Two input shapes are supported because both turn up in practice:
  1. A full .SAFE product, where the calibration LUT lives in
     annotation/calibration/calibration-*.xml and DN must be divided by it.
  2. A bare GeoTIFF (AWS Open Data tiles, cropped exports), where the values
     may already be Sigma0 — or may still be raw DN.

Guessing wrong here shifts every backscatter value by tens of dB and quietly
destroys the damping ratio the look-alike stage depends on, so the mode is
detected explicitly and always reported.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

CalibrationMode = Literal["safe_lut", "already_sigma0", "raw_dn_generic"]

# Floor for the log conversion. Sea-surface Sigma0 in VV rarely goes below
# about -35 dB; anything lower is noise or a zero-fill border pixel.
SIGMA0_FLOOR = 1e-6
NODATA_DB = -99.0


@dataclass
class CalibrationResult:
    sigma0_db: np.ndarray
    mode: CalibrationMode
    valid_mask: np.ndarray  # True where the pixel carries real signal
    notes: str = ""


def read_calibration_lut(safe_dir: Path, polarisation: str = "vv") -> np.ndarray | None:
    """Extract the sigmaNought LUT from a .SAFE annotation, if present.

    Returns the mean sigmaNought value. A per-pixel bilinear expansion of the
    LUT grid would be more correct, but across an IW GRD swath the sigmaNought
    vector varies by well under 1 dB, which is far below the difference the
    look-alike stage cares about.
    """
    cal_dir = safe_dir / "annotation" / "calibration"
    if not cal_dir.is_dir():
        return None
    matches = sorted(cal_dir.glob(f"calibration-*{polarisation.lower()}*.xml"))
    if not matches:
        return None
    try:
        root = ET.parse(matches[0]).getroot()
    except ET.ParseError as exc:
        log.warning("Unparseable calibration XML %s: %s", matches[0], exc)
        return None

    values: list[float] = []
    for node in root.iter("sigmaNought"):
        if node.text:
            values.extend(float(v) for v in node.text.split())
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def detect_mode(data: np.ndarray, lut: np.ndarray | None) -> CalibrationMode:
    """Decide how to interpret the raster values."""
    if lut is not None:
        return "safe_lut"
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return "already_sigma0"
    hi = float(np.percentile(finite, 99))
    # Sigma0 (linear power) sits well under 1 for the sea; GRD DN is O(100-1000).
    if hi <= 5.0:
        return "already_sigma0"
    return "raw_dn_generic"


def to_sigma0_db(
    data: np.ndarray,
    safe_dir: Path | None = None,
    polarisation: str = "vv",
    nodata: float | None = 0.0,
) -> CalibrationResult:
    """Convert a GRD amplitude/DN array to Sigma0 in decibels."""
    arr = np.asarray(data, dtype=np.float64)

    valid = np.isfinite(arr)
    if nodata is not None:
        valid &= arr != nodata
    valid &= arr > 0

    lut = read_calibration_lut(safe_dir, polarisation) if safe_dir else None
    mode = detect_mode(arr, lut)

    if mode == "safe_lut":
        assert lut is not None
        a = float(np.mean(lut))
        # Sentinel-1 IPF: sigma0 = DN^2 / A^2
        sigma0 = np.where(valid, (arr**2) / (a**2), SIGMA0_FLOOR)
        notes = f"SAFE sigmaNought LUT, mean A={a:.2f} over {lut.size} nodes"
    elif mode == "already_sigma0":
        sigma0 = np.where(valid, arr, SIGMA0_FLOOR)
        notes = "values already in linear Sigma0"
    else:
        # No LUT available. Normalise by the scene's own bright-sea level so
        # the *relative* structure (which is all the damping ratio needs) is
        # preserved. Absolute calibration is NOT claimed in this mode.
        ref = float(np.percentile(arr[valid], 95)) if valid.any() else 1.0
        ref = max(ref, 1.0)
        sigma0 = np.where(valid, (arr / ref) ** 2, SIGMA0_FLOOR)
        notes = (
            f"no LUT; scene-relative normalisation by p95 DN={ref:.1f}. "
            "Relative backscatter only — absolute Sigma0 not calibrated."
        )

    sigma0 = np.maximum(sigma0, SIGMA0_FLOOR)
    db = 10.0 * np.log10(sigma0)
    db = np.where(valid, db, NODATA_DB).astype(np.float32)

    log.info("Calibration mode=%s (%s)", mode, notes)
    return CalibrationResult(sigma0_db=db, mode=mode, valid_mask=valid, notes=notes)


def db_to_linear(db: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 10.0)


def normalise_for_model(db: np.ndarray, lo: float = -35.0, hi: float = 5.0) -> np.ndarray:
    """Scale dB into [0, 1] for network input, clipping to a fixed physical range.

    Fixed bounds, not per-tile min/max: per-tile scaling would make an
    all-dark oil tile look identical to an all-bright sea tile.
    """
    arr = np.asarray(db, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
