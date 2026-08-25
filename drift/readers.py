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
    """Reads u/v from a CMEMS or ERA5 NetCDF.

    Variable names differ between products, so they are passed in rather than
    guessed. Interpolation is linear in space and time; requests outside the
    file's coverage raise instead of silently clamping to the nearest edge,
    because a drift run quietly pinned to the domain boundary produces a
    confident and completely wrong origin.

    The whole field is loaded into numpy once and interpolated by hand.
    xarray.interp is convenient but it realigns dimensions and builds an
    intermediate Dataset on every call, and a drift run calls this once per
    particle per timestep - 300 particles over 24 steps for each of several
    candidates is tens of thousands of calls. That put one real-currents scene
    at 620 s against a two-minute whole-scene target. A CMEMS subset for one
    scene is only a few million floats, so it fits in memory comfortably and
    each sample becomes a handful of array lookups.
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
        import numpy as np
        import xarray as xr

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"NetCDF field not found: {self.path}")

        self.u_var, self.v_var = u_var, v_var
        self.lon_var, self.lat_var, self.time_var = lon_var, lat_var, time_var
        self.name = name
        self.source = f"{name}:{self.path.name}"

        with xr.open_dataset(self.path) as ds:
            for var in (u_var, v_var):
                if var not in ds:
                    raise KeyError(
                        f"{var!r} not in {self.path.name}; "
                        f"available: {list(ds.data_vars)}"
                    )
            u, v = ds[u_var], ds[v_var]
            if depth_index is not None and "depth" in u.dims:
                u, v = u.isel(depth=depth_index), v.isel(depth=depth_index)
            u, v = u.squeeze(), v.squeeze()

            self._lons = np.asarray(ds[lon_var].values, dtype=np.float64)
            self._lats = np.asarray(ds[lat_var].values, dtype=np.float64)
            self._has_time = time_var in u.dims
            order = [d for d in (time_var, lat_var, lon_var) if d in u.dims]
            u = u.transpose(*order)
            v = v.transpose(*order)
            self._u = np.asarray(u.values, dtype=np.float64)
            self._v = np.asarray(v.values, dtype=np.float64)
            if self._has_time:
                # int64 nanoseconds: monotonic, exact, and cheap to search.
                self._times = ds[time_var].values.astype("datetime64[ns]").astype(np.int64)
            else:
                self._times = None

        # searchsorted needs ascending axes. Latitude is descending in some
        # products (ERA5 among them), so flip rather than assume.
        if self._lats.size > 1 and self._lats[0] > self._lats[-1]:
            self._lats = self._lats[::-1]
            axis = 1 if self._has_time else 0
            self._u = np.flip(self._u, axis=axis)
            self._v = np.flip(self._v, axis=axis)
        if self._lons.size > 1 and self._lons[0] > self._lons[-1]:
            self._lons = self._lons[::-1]
            axis = 2 if self._has_time else 1
            self._u = np.flip(self._u, axis=axis)
            self._v = np.flip(self._v, axis=axis)

    @staticmethod
    def _bracket(axis, value: float, label: str, unit: str = ""):
        """Lower index and weight for `value` on an ascending axis.

        Raises outside the range rather than clamping - see the class docstring.
        """
        import numpy as np

        if axis.size == 1:
            if not np.isclose(float(axis[0]), value, atol=1e-6):
                raise ValueError(
                    f"{label} {value}{unit} is outside the single {label} "
                    f"value {axis[0]}{unit} in this field"
                )
            return 0, 0.0
        if value < axis[0] or value > axis[-1]:
            raise ValueError(
                f"{label} {value}{unit} is outside the field's range "
                f"{axis[0]}{unit} to {axis[-1]}{unit}"
            )
        i = int(np.searchsorted(axis, value)) - 1
        i = min(max(i, 0), axis.size - 2)
        span = axis[i + 1] - axis[i]
        weight = 0.0 if span == 0 else float((value - axis[i]) / span)
        return i, weight

    def sample(self, lon: float, lat: float, when: datetime) -> FieldSample:
        import numpy as np

        try:
            j, wy = self._bracket(self._lats, lat, "latitude", " deg")
            i, wx = self._bracket(self._lons, lon, "longitude", " deg")
        except ValueError as exc:
            raise ValueError(
                f"Could not interpolate {self.path.name} at "
                f"({lon:.3f}, {lat:.3f}, {when}): {exc}"
            ) from exc

        def plane(cube, k: int):
            """Bilinear blend of one time slice."""
            block = cube[k] if self._has_time else cube
            a = block[j, i]; b = block[j, i + 1]
            c = block[j + 1, i]; d = block[j + 1, i + 1]
            return ((a * (1 - wx) + b * wx) * (1 - wy)
                    + (c * (1 - wx) + d * wx) * wy)

        if self._has_time and self._times is not None:
            stamp = np.datetime64(when.replace(tzinfo=None), "ns").astype(np.int64)
            try:
                k, wt = self._bracket(self._times, float(stamp), "time")
            except ValueError as exc:
                raise ValueError(
                    f"Could not interpolate {self.path.name} at "
                    f"({lon:.3f}, {lat:.3f}, {when}): {exc}"
                ) from exc
            uu = plane(self._u, k) * (1 - wt) + plane(self._u, k + 1) * wt
            vv = plane(self._v, k) * (1 - wt) + plane(self._v, k + 1) * wt
        else:
            uu, vv = plane(self._u, 0), plane(self._v, 0)

        uu, vv = float(uu), float(vv)
        if not (math.isfinite(uu) and math.isfinite(vv)):
            raise ValueError(
                f"{self.name} returned NaN at ({lon:.3f}, {lat:.3f}, {when}) - "
                f"point is outside the model domain or on land"
            )
        return FieldSample(uu, vv, self.source)

    def close(self) -> None:
        """No-op: the file is read once at construction and closed there."""
        return None


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
