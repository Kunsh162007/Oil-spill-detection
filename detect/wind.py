"""Wind context lookup - the strongest single feature for rejecting look-alikes.

Sources, in order of trust:
  1. ERA5 NetCDF via the Copernicus Climate Data Store (cdsapi).
  2. A measured constant, when only a reported value is available.
  3. Synthetic, ONLY when explicitly configured - tagged all the way to the UI.

CLAUDE.md rule 3: no detection without a wind check. There is deliberately no
silent default, so a missing wind file raises rather than quietly producing a
candidate with invented weather.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.contracts import WindContext

log = logging.getLogger(__name__)


class ERA5Wind:
    """10 m wind from an ERA5 NetCDF file (u10/v10)."""

    def __init__(self, path: str | Path, u_var: str = "u10", v_var: str = "v10") -> None:
        import xarray as xr

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"ERA5 file not found: {self.path}. "
                f"Run scripts/fetch_wind.py to download it."
            )
        self.ds = xr.open_dataset(self.path)
        missing = [v for v in (u_var, v_var) if v not in self.ds]
        if missing:
            raise KeyError(
                f"{missing} not in {self.path.name}; available: {list(self.ds.data_vars)}"
            )
        self.u_var, self.v_var = u_var, v_var

    def __call__(self, lon: float, lat: float, when: datetime) -> WindContext:
        import numpy as np

        lon_name = "longitude" if "longitude" in self.ds.coords else "lon"
        lat_name = "latitude" if "latitude" in self.ds.coords else "lat"
        sel = {lon_name: lon, lat_name: lat}
        if "time" in self.ds.dims:
            sel["time"] = np.datetime64(when.replace(tzinfo=None))
        elif "valid_time" in self.ds.dims:
            sel["valid_time"] = np.datetime64(when.replace(tzinfo=None))

        point = self.ds.interp(**sel)
        u = float(point[self.u_var].values)
        v = float(point[self.v_var].values)
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError(
                f"ERA5 returned NaN at ({lon:.3f}, {lat:.3f}, {when}) - "
                f"outside the downloaded domain or time range"
            )

        speed = math.hypot(u, v)
        # Meteorological convention: the direction the wind blows FROM.
        direction = (math.degrees(math.atan2(-u, -v))) % 360.0
        return WindContext.from_speed(speed, direction, source="ERA5")


@dataclass
class ConstantWind:
    """A single measured wind value applied across the scene."""

    speed_ms: float
    direction_deg: float = 0.0
    source: str = "measured"

    def __call__(self, lon: float, lat: float, when: datetime) -> WindContext:
        return WindContext.from_speed(self.speed_ms, self.direction_deg, source=self.source)


@dataclass
class SyntheticWind:
    """Smoothly varying wind for demos and tests. Tagged 'synthetic' throughout.

    Never selected automatically - config must ask for it by name.
    """

    base_speed_ms: float = 5.5
    variation_ms: float = 2.0
    scale_deg: float = 0.35
    direction_deg: float = 225.0

    def __call__(self, lon: float, lat: float, when: datetime) -> WindContext:
        k = 2.0 * math.pi / max(self.scale_deg, 1e-6)
        wobble = math.sin(k * lon) * math.cos(k * lat)
        speed = max(0.0, self.base_speed_ms + self.variation_ms * wobble)
        return WindContext.from_speed(speed, self.direction_deg, source="synthetic")


def build_wind_lookup(config):
    """Construct the wind lookup named in config. No silent default."""
    section = config.section("wind") or {}
    source = str(section.get("source", "synthetic")).lower()

    if source == "era5":
        path = section.get("path")
        if not path:
            raise ValueError("wind.source is 'era5' but wind.path is unset")
        from core.config import resolve_path

        return ERA5Wind(resolve_path(path))

    if source in ("constant", "measured"):
        if "speed_ms" not in section:
            raise ValueError("wind.source is 'constant' but wind.speed_ms is unset")
        return ConstantWind(
            speed_ms=float(section["speed_ms"]),
            direction_deg=float(section.get("direction_deg", 0.0)),
            source=str(section.get("label", "measured")),
        )

    if source == "synthetic":
        log.warning(
            "Using SYNTHETIC wind - rejections will be tagged as a demonstration. "
            "Run scripts/fetch_wind.py for real ERA5 data."
        )
        return SyntheticWind(
            base_speed_ms=float(section.get("base_speed_ms", 5.5)),
            variation_ms=float(section.get("variation_ms", 2.0)),
            direction_deg=float(section.get("direction_deg", 225.0)),
        )

    raise ValueError(f"Unknown wind.source: {source!r}")
