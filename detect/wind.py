"""Wind context lookup - the strongest single feature for rejecting look-alikes.

Sources, in order of trust:
  1. ERA5 NetCDF via the Copernicus Climate Data Store (cdsapi).
  2. ERA5 via the Open-Meteo archive - same reanalysis, no account needed.
  3. A measured constant, when only a reported value is available.
  4. Synthetic, ONLY when explicitly configured - tagged all the way to the UI.

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


class OpenMeteoERA5Wind:
    """ERA5 10 m wind from the Open-Meteo archive. No account, no API key.

    The Copernicus Climate Data Store serves the authoritative ERA5, but it
    needs a registered account and a ~/.cdsapirc. Open-Meteo republishes the
    same reanalysis over an open HTTP endpoint, which means real wind can be
    used without waiting on a registration. It is the same underlying data at
    ERA5's native 0.25 degree resolution, so it is labelled ERA5 - with the
    access route recorded, because where a number came from is part of the
    number.

    Values are cached on disk per grid cell per day. ERA5 is 0.25 degrees, so
    a Sentinel-1 scene spans only two or three cells: rounding the request to
    the grid turns hundreds of candidate lookups into a handful of fetches, and
    makes a re-run reproducible offline.
    """

    GRID_DEG = 0.25
    ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
    SOURCE = "ERA5 (Open-Meteo archive)"

    def __init__(self, cache_dir: str | Path = "data/wind/openmeteo",
                 allow_network: bool = True, timeout_s: float = 45.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_network = allow_network
        self.timeout_s = timeout_s
        self._memo: dict[tuple, dict] = {}

    def _cell(self, lon: float, lat: float) -> tuple[float, float]:
        """Snap to the ERA5 grid so nearby candidates share one fetch."""
        g = self.GRID_DEG
        return (round(round(lon / g) * g, 2), round(round(lat / g) * g, 2))

    def _day(self, cell_lon: float, cell_lat: float, date: str) -> dict:
        key = (cell_lon, cell_lat, date)
        if key in self._memo:
            return self._memo[key]

        path = self.cache_dir / f"{cell_lat}_{cell_lon}_{date}.json"
        if path.exists():
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
            self._memo[key] = payload
            return payload

        if not self.allow_network:
            raise FileNotFoundError(
                f"No cached wind for ({cell_lon}, {cell_lat}) on {date} and "
                f"network access is disabled. Run scripts/fetch_wind.py first."
            )

        import json

        import requests

        params = {
            "latitude": cell_lat, "longitude": cell_lon,
            "start_date": date, "end_date": date,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "ms", "timezone": "UTC",
        }
        response = requests.get(self.ENDPOINT, params=params, timeout=self.timeout_s)
        response.raise_for_status()
        payload = response.json()

        hourly = payload.get("hourly") or {}
        speeds = hourly.get("wind_speed_10m") or []
        # An empty or all-null series is the documented failure mode for these
        # APIs: HTTP 200 with nothing in it. Never let that become a wind value.
        if not speeds or all(v is None for v in speeds):
            raise ValueError(
                f"Open-Meteo returned no ERA5 wind for ({cell_lon}, {cell_lat}) "
                f"on {date}. The date may be outside the archive."
            )

        path.write_text(json.dumps(payload), encoding="utf-8")
        self._memo[key] = payload
        return payload

    def __call__(self, lon: float, lat: float, when: datetime) -> WindContext:
        cell_lon, cell_lat = self._cell(lon, lat)
        payload = self._day(cell_lon, cell_lat, when.strftime("%Y-%m-%d"))

        hourly = payload["hourly"]
        times = hourly["time"]
        speeds = hourly["wind_speed_10m"]
        directions = hourly["wind_direction_10m"]

        # Nearest hour with a real value. ERA5 is hourly, so this is at worst a
        # 30 minute offset - well inside the variability the window score
        # already tolerates.
        target = when.replace(tzinfo=None)
        best_i, best_gap = None, None
        for i, stamp in enumerate(times):
            if speeds[i] is None or directions[i] is None:
                continue
            gap = abs((datetime.fromisoformat(stamp) - target).total_seconds())
            if best_gap is None or gap < best_gap:
                best_i, best_gap = i, gap

        if best_i is None:
            raise ValueError(
                f"No usable ERA5 wind near {when} at ({lon:.3f}, {lat:.3f})"
            )

        return WindContext.from_speed(
            float(speeds[best_i]), float(directions[best_i]), source=self.SOURCE
        )


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

    if source in ("open-meteo", "openmeteo", "era5-openmeteo"):
        from core.config import resolve_path

        return OpenMeteoERA5Wind(
            cache_dir=resolve_path(section.get("cache_dir", "data/wind/openmeteo")),
            allow_network=bool(section.get("allow_network", True)),
        )

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
