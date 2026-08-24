"""Environmental field providers for the drift model.

Three sources, in descending order of trust:

  1. CMEMS / ERA5 NetCDF - the real thing.
  2. A constant field measured from observations, when only a summary value
     is known (e.g. a reported current at the scene centre).
  3. A synthetic field - ONLY when explicitly requested in config.

Rule 6 of CLAUDE.md forbids inventing data. So the synthetic reader is never
selected automatically: it must be asked for by name, and everything it
produces is tagged source="synthetic" all the way to the UI, which labels it.
A drift origin computed on invented currents is a demo prop, not a result,
and it has to look like one on screen.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass
class FieldSample:
    """Eastward/northward components at one point and time."""

    u: float  # m/s, positive east
    v: float  # m/s, positive north
    source: str

    @property
    def speed(self) -> float:
        return math.hypot(self.u, self.v)

    @property
    def direction_deg(self) -> float:
        """Direction the flow is going TOWARDS, degrees from north."""
        return math.degrees(math.atan2(self.u, self.v)) % 360.0


class VectorField(Protocol):
    """Anything that can report u/v at a point in space and time."""

    name: str

    def sample(self, lon: float, lat: float, when: datetime) -> FieldSample: ...


@dataclass
class ConstantField:
    """A uniform, time-invariant field from a known measured value."""

    u: float
    v: float
    name: str = "constant"
    source: str = "measured-constant"

    def sample(self, lon: float, lat: float, when: datetime) -> FieldSample:
        return FieldSample(self.u, self.v, self.source)

    @classmethod
    def from_speed_direction(
        cls, speed_ms: float, direction_deg: float, source: str = "measured-constant"
    ) -> ConstantField:
        """Build from speed and the direction the flow moves TOWARDS."""
        rad = math.radians(direction_deg)
        return cls(u=speed_ms * math.sin(rad), v=speed_ms * math.cos(rad), source=source)


class NetCDFField:
    """Reads u/v from a CMEMS or ERA5 NetCDF via xarray.

    Variable names differ between products, so they are passed in rather than
    guessed. Interpolation is linear in space and time; requests outside the
    file's coverage raise instead of silently clamping to the nearest edge,
    because a drift run quietly pinned to the domain boundary produces a
    confident and completely wrong origin.
    """

    def __init__(
        self,
        path: str | Path,
        u_var: str = "uo",
        v_var: str = "vo",
        lon_var: str = "longitude",
        lat_var: str = "latitude",
        time_var: str = "time",
        depth_index: int | None = 0,
        name: str = "cmems",
    ) -> None:
        import xarray as xr

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"NetCDF field not found: {self.path}")
        self.ds = xr.open_dataset(self.path)
        for var in (u_var, v_var):
            if var not in self.ds:
                raise KeyError(
                    f"{var!r} not in {self.path.name}; available: {list(self.ds.data_vars)}"
                )
        self.u_var, self.v_var = u_var, v_var
        self.lon_var, self.lat_var, self.time_var = lon_var, lat_var, time_var
        self.depth_index = depth_index
        self.name = name
        self.source = f"{name}:{self.path.name}"

    def sample(self, lon: float, lat: float, when: datetime) -> FieldSample:
        import numpy as np

        sel = {self.lon_var: lon, self.lat_var: lat}
        if self.time_var in self.ds.dims:
            sel[self.time_var] = np.datetime64(when.replace(tzinfo=None))

        try:
            point = self.ds.interp(**sel)
        except Exception as exc:
            raise ValueError(
                f"Could not interpolate {self.path.name} at ({lon:.3f}, {lat:.3f}, {when}): {exc}"
            ) from exc

        u = point[self.u_var]
        v = point[self.v_var]
        if self.depth_index is not None and "depth" in u.dims:
            u = u.isel(depth=self.depth_index)
            v = v.isel(depth=self.depth_index)

        uu, vv = float(u.values), float(v.values)
        if not (math.isfinite(uu) and math.isfinite(vv)):
            raise ValueError(
                f"{self.name} returned NaN at ({lon:.3f}, {lat:.3f}, {when}) - "
                f"point is outside the model domain or on land"
            )
        return FieldSample(uu, vv, self.source)

    def close(self) -> None:
        self.ds.close()


@dataclass
class SyntheticField:
    """A smooth, plausible current field for demos and tests.

    NEVER selected automatically. It exists so the pipeline can be exercised
    end to end without CMEMS credentials, and so tests are deterministic.
    Its source tag is "synthetic" and every downstream surface must show that
    to the viewer - see decision/ and the UI banner.
    """

    base_u: float = 0.25
    base_v: float = -0.10
    eddy_scale_deg: float = 0.6
    eddy_strength: float = 0.18
    tidal_amplitude: float = 0.12
    tidal_period_h: float = 12.42  # M2 semidiurnal
    name: str = "synthetic"
    source: str = "synthetic"

    def sample(self, lon: float, lat: float, when: datetime) -> FieldSample:
        k = 2.0 * math.pi / max(self.eddy_scale_deg, 1e-6)
        # Divergence-free eddy pattern from a streamfunction, so the field
        # neither creates nor destroys water - particles behave sensibly.
        u_eddy = self.eddy_strength * math.sin(k * lon) * math.cos(k * lat)
        v_eddy = -self.eddy_strength * math.cos(k * lon) * math.sin(k * lat)

        hours = when.timestamp() / 3600.0
        phase = 2.0 * math.pi * hours / self.tidal_period_h
        u_tide = self.tidal_amplitude * math.cos(phase)
        v_tide = self.tidal_amplitude * 0.4 * math.sin(phase)

        return FieldSample(
            u=self.base_u + u_eddy + u_tide,
            v=self.base_v + v_eddy + v_tide,
            source=self.source,
        )


@dataclass
class WindField:
    """Wraps a vector field as a wind source and applies the drift factor.

    Surface oil moves at roughly 3% of the wind speed. That factor is applied
    here rather than in the integrator so the number appears exactly once.
    """

    field: VectorField
    drift_factor: float = 0.03

    @property
    def name(self) -> str:
        return getattr(self.field, "name", "wind")

    def sample(self, lon: float, lat: float, when: datetime) -> FieldSample:
        s = self.field.sample(lon, lat, when)
        return FieldSample(
            s.u * self.drift_factor, s.v * self.drift_factor, f"{s.source}x{self.drift_factor}"
        )


def zero_field(name: str = "none") -> ConstantField:
    """A field of no motion. Used when wind is genuinely unavailable."""
    return ConstantField(0.0, 0.0, name=name, source=f"{name}-zero")
