"""Forward drift - where is the slick NOW?

Backward drift answers "where did this start". This answers the operational
question: the image is 3-24 hours old by the time we see it, and a response
vessel needs the slick's present position, not its position at acquisition.

Forward is the physically natural direction, so unlike the backward run this
one COULD include weathering. We still leave it out and say so: modelling
evaporation and emulsification without oil-type and thickness inputs - which
SAR cannot measure - would add false precision, not accuracy. What we model
is advection, and the uncertainty grows accordingly.

Three states the UI shows on one timeline:

    origin  ──(backward)──  observed  ──(forward)──  now
    where it started        what SAR saw            where it probably is
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

from core.geo import haversine_km
from drift.backward import (
    MODEL_ERROR_KM_PER_HOUR,
    _cloud_stats,
    _rk4_step,
    seed_particles,
)
from drift.readers import VectorField, zero_field

log = logging.getLogger(__name__)

# Beyond this the forward estimate is a search area, not a position.
FORECAST_RELIABLE_HOURS = 48.0

# Hard ceiling. Advecting a slick for weeks produces a confident-looking
# position on the far side of an ocean, which is worse than no answer: real
# oil disperses, strands or weathers away long before that. Past this age a
# detection is HISTORICAL and gets no present-position estimate at all.
MAX_FORECAST_HOURS = 72.0


@dataclass
class DriftState:
    """The slick at one moment in time."""

    label: str                     # "origin" | "observed" | "now"
    lon: float
    lat: float
    at: datetime
    uncertainty_km: float
    description: str
    hours_from_observation: float = 0.0
    area_km2: float | None = None
    polygon: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "lon": round(self.lon, 6),
            "lat": round(self.lat, 6),
            "at": self.at.isoformat(),
            "uncertainty_km": round(self.uncertainty_km, 2),
            "description": self.description,
            "hours_from_observation": round(self.hours_from_observation, 2),
            "area_km2": round(self.area_km2, 3) if self.area_km2 else None,
            "polygon": self.polygon,
        }


@dataclass
class ForwardResult:
    state: DriftState
    track: list[dict[str, Any]]
    particles_lonlat: list[tuple[float, float]]
    reliable: bool
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _hull_polygon(points: list[tuple[float, float]]) -> list[list[float]]:
    """Convex hull of the particle cloud - the drifted slick's likely footprint."""
    if len(points) < 3:
        return []
    try:
        from shapely.geometry import MultiPoint

        hull = MultiPoint(points).convex_hull
        if hull.geom_type != "Polygon":
            return []
        return [[round(float(x), 6), round(float(y), 6)] for x, y in hull.exterior.coords]
    except Exception:
        return []


def forward_drift(
    polygon_lonlat: Sequence[tuple[float, float]],
    observed_at: datetime,
    until: datetime | None,
    currents: VectorField,
    wind: VectorField | None = None,
    timestep_minutes: float = 30.0,
    n_particles: int = 300,
    diffusion_m2_s: float = 5.0,
    seed: int = 0,
) -> ForwardResult:
    """Advect the observed slick forward to `until` (default: now).

    Uses the same integrator and seeding as the backward run, so the two
    directions are directly comparable rather than two different models.
    """
    until = until or datetime.now(timezone.utc)
    hours = (until - observed_at).total_seconds() / 3600.0

    rng = np.random.default_rng(seed)
    wind = wind or zero_field("no-wind")
    particles = seed_particles(polygon_lonlat, n_particles, rng)
    warnings: list[str] = []

    if hours > MAX_FORECAST_HOURS:
        # Refuse rather than extrapolate. See MAX_FORECAST_HOURS.
        lon, lat, spread = _cloud_stats(particles)
        return ForwardResult(
            state=DriftState(
                label="historical", lon=lon, lat=lat, at=observed_at,
                uncertainty_km=spread, hours_from_observation=hours,
                description=(
                    f"acquired {hours / 24.0:.0f} days ago - too old to forecast a "
                    f"present position. Oil disperses, strands or weathers away "
                    f"well within that time."
                ),
                polygon=_hull_polygon(particles),
            ),
            track=[], particles_lonlat=particles, reliable=False,
            warnings=[
                f"no present-position estimate: the scene is {hours / 24.0:.0f} days "
                f"old, beyond the {MAX_FORECAST_HOURS:.0f} h forecast ceiling"
            ],
            stats={"hours": round(hours, 2), "historical": True},
        )

    if hours <= 0:
        # The scene is from the future relative to `until`; nothing to do.
        lon, lat, spread = _cloud_stats(particles)
        return ForwardResult(
            state=DriftState(
                label="now", lon=lon, lat=lat, at=observed_at,
                uncertainty_km=spread,
                description="observation is not in the past; no forward drift applied",
                polygon=_hull_polygon(particles),
            ),
            track=[], particles_lonlat=particles, reliable=True,
            warnings=["forward drift skipped: target time precedes acquisition"],
        )

    dt_s = abs(timestep_minutes) * 60.0          # positive: forwards
    n_steps = max(1, int(round(hours * 60.0 / timestep_minutes)))
    diff_sigma_m = math.sqrt(2.0 * diffusion_m2_s * dt_s)

    from core.geo import offset_by_metres

    when = observed_at
    track: list[dict[str, Any]] = []
    lon, lat, spread = _cloud_stats(particles)
    track.append({
        "lon": round(lon, 6), "lat": round(lat, 6),
        "at": when.isoformat(), "hours_after": 0.0,
        "spread_km": round(spread, 3),
    })

    for step in range(1, n_steps + 1):
        moved: list[tuple[float, float]] = []
        for plon, plat in particles:
            try:
                nlon, nlat = _rk4_step(plon, plat, when, dt_s, currents, wind)
            except ValueError as exc:
                msg = f"particle left the field domain at step {step}: {exc}"
                if msg not in warnings:
                    warnings.append(msg)
                nlon, nlat = plon, plat
            if diff_sigma_m > 0:
                nlon, nlat = offset_by_metres(
                    (nlon, nlat),
                    float(rng.normal(0.0, diff_sigma_m)),
                    float(rng.normal(0.0, diff_sigma_m)),
                )
            moved.append((nlon, nlat))
        particles = moved
        when = when + timedelta(seconds=dt_s)
        lon, lat, spread = _cloud_stats(particles)
        track.append({
            "lon": round(lon, 6), "lat": round(lat, 6),
            "at": when.isoformat(),
            "hours_after": round(step * timestep_minutes / 60.0, 3),
            "spread_km": round(spread, 3),
        })

    model_error = MODEL_ERROR_KM_PER_HOUR * hours
    uncertainty = math.sqrt(spread**2 + model_error**2)
    reliable = hours <= FORECAST_RELIABLE_HOURS

    if not reliable:
        warnings.append(
            f"forecast horizon {hours:.0f} h exceeds {FORECAST_RELIABLE_HOURS:.0f} h - "
            f"treat this as a search area, not a position"
        )
    warnings.append(
        "advection only: weathering is not modelled, because SAR cannot measure "
        "the oil thickness or type those processes depend on"
    )

    displacement = haversine_km(
        (track[0]["lon"], track[0]["lat"]), (lon, lat)
    )
    log.info(
        "Forward drift %.1f h -> (%.4f, %.4f), moved %.1f km, +/-%.1f km",
        hours, lon, lat, displacement, uncertainty,
    )

    return ForwardResult(
        state=DriftState(
            label="now", lon=lon, lat=lat, at=when,
            uncertainty_km=uncertainty,
            hours_from_observation=hours,
            description=(
                f"estimated present position, {hours:.1f} h after acquisition, "
                f"having drifted about {displacement:.1f} km"
            ),
            polygon=_hull_polygon(particles),
        ),
        track=track,
        particles_lonlat=particles,
        reliable=reliable,
        warnings=warnings,
        stats={
            "hours": round(hours, 2),
            "displacement_km": round(displacement, 2),
            "particle_spread_km": round(spread, 2),
            "model_error_km": round(model_error, 2),
            "n_steps": n_steps,
        },
    )
