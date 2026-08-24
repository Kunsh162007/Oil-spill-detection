"""Backward drift - rewind a slick to where it started.

Seed particles across the observed polygon, integrate the velocity field
BACKWARDS in time, and report where the cloud converges plus how tightly.

Two things are deliberately not done here:

  * Weathering is disabled. Evaporation, emulsification and dispersion are
    irreversible; running them backwards is not physics, it is arithmetic
    that happens to produce numbers. OpenOil is configured accordingly.
  * Backward drift is NOT forward drift with a minus sign for nonlinear
    processes. We integrate advection only, and the uncertainty term below
    grows with backtrack time to reflect exactly that.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np

from core.contracts import DriftOrigin
from core.geo import haversine_km, offset_by_metres
from drift.readers import FieldSample, VectorField, zero_field

log = logging.getLogger(__name__)

# Advection error accumulates roughly linearly in ocean models; this is a
# conservative per-hour term added to the particle spread so the reported
# uncertainty does not pretend the current field is perfect.
MODEL_ERROR_KM_PER_HOUR = 0.9


@dataclass
class DriftResult:
    origin: DriftOrigin
    particles_lonlat: list[tuple[float, float]]
    steps: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def seed_particles(
    polygon_lonlat: Sequence[tuple[float, float]],
    n_particles: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Scatter particles uniformly inside the slick polygon.

    Rejection sampling inside the bounding box. Seeding across the whole
    polygon rather than at its centroid is what lets the spread of the
    back-tracked cloud carry real information about origin uncertainty.
    """
    pts = [(float(x), float(y)) for x, y in polygon_lonlat]
    if len(pts) < 3:
        raise ValueError("Need at least 3 polygon vertices to seed particles")

    from shapely.geometry import Point, Polygon

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        raise ValueError("Slick polygon has no area - cannot seed drift particles")

    min_lon, min_lat, max_lon, max_lat = poly.bounds
    out: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = n_particles * 200

    while len(out) < n_particles and attempts < max_attempts:
        lon = rng.uniform(min_lon, max_lon)
        lat = rng.uniform(min_lat, max_lat)
        attempts += 1
        if poly.contains(Point(lon, lat)):
            out.append((lon, lat))

    if not out:
        raise ValueError("Failed to seed any particle inside the slick polygon")
    if len(out) < n_particles:
        log.warning(
            "Seeded only %d/%d particles (thin polygon); continuing",
            len(out), n_particles,
        )
    return out


def _velocity(
    lon: float, lat: float, when: datetime,
    currents: VectorField, wind: VectorField,
) -> tuple[float, float]:
    """Total surface velocity: currents plus the wind-driven component."""
    c = currents.sample(lon, lat, when)
    w = wind.sample(lon, lat, when)
    return c.u + w.u, c.v + w.v


def _rk4_step(
    lon: float, lat: float, when: datetime, dt_s: float,
    currents: VectorField, wind: VectorField,
) -> tuple[float, float]:
    """One RK4 advection step. dt_s is NEGATIVE when running backwards.

    RK4 rather than Euler because a 30-minute step through an eddy field
    accumulates visible along-track error with a first-order scheme, and the
    whole point of this stage is where the track ends up.
    """
    half = timedelta(seconds=dt_s / 2.0)
    full = timedelta(seconds=dt_s)

    u1, v1 = _velocity(lon, lat, when, currents, wind)
    p2 = offset_by_metres((lon, lat), u1 * dt_s / 2.0, v1 * dt_s / 2.0)
    u2, v2 = _velocity(p2[0], p2[1], when + half, currents, wind)
    p3 = offset_by_metres((lon, lat), u2 * dt_s / 2.0, v2 * dt_s / 2.0)
    u3, v3 = _velocity(p3[0], p3[1], when + half, currents, wind)
    p4 = offset_by_metres((lon, lat), u3 * dt_s, v3 * dt_s)
    u4, v4 = _velocity(p4[0], p4[1], when + full, currents, wind)

    u = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0
    v = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0
    return offset_by_metres((lon, lat), u * dt_s, v * dt_s)


def _cloud_stats(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Centroid (lon, lat) and the radius containing ~68% of particles, km."""
    lons = np.array([p[0] for p in points], dtype=np.float64)
    lats = np.array([p[1] for p in points], dtype=np.float64)
    c_lon, c_lat = float(lons.mean()), float(lats.mean())
    dists = np.array(
        [haversine_km((c_lon, c_lat), (lo, la)) for lo, la in points], dtype=np.float64
    )
    spread = float(np.percentile(dists, 68)) if dists.size else 0.0
    return c_lon, c_lat, spread


def backtrack(
    polygon_lonlat: Sequence[tuple[float, float]],
    observed_at: datetime,
    currents: VectorField,
    wind: VectorField | None = None,
    backtrack_hours: float = 12.0,
    timestep_minutes: float = 30.0,
    n_particles: int = 500,
    diffusion_m2_s: float = 5.0,
    seed: int = 0,
) -> DriftResult:
    """Run particles backwards from an observed slick to a probable origin."""
    if backtrack_hours <= 0:
        raise ValueError("backtrack_hours must be positive")

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    wind = wind or zero_field("no-wind")

    particles = seed_particles(polygon_lonlat, n_particles, rng)
    dt_s = -abs(timestep_minutes) * 60.0          # negative: backwards
    n_steps = max(1, int(round(backtrack_hours * 60.0 / timestep_minutes)))
    # Random-walk step for horizontal eddy diffusivity K: sigma = sqrt(2*K*dt)
    diff_sigma_m = math.sqrt(2.0 * diffusion_m2_s * abs(dt_s))

    warnings: list[str] = []
    steps: list[dict[str, Any]] = []
    when = observed_at

    c_lon, c_lat, spread = _cloud_stats(particles)
    steps.append({
        "step": 0,
        "at": observed_at,
        "hours_before": 0.0,
        "lon": c_lon, "lat": c_lat,
        "spread_km": round(spread, 3),
    })

    for step in range(1, n_steps + 1):
        moved: list[tuple[float, float]] = []
        for lon, lat in particles:
            try:
                nlon, nlat = _rk4_step(lon, lat, when, dt_s, currents, wind)
            except ValueError as exc:
                # A particle leaving the model domain is information, not a
                # crash: park it and record that coverage ran out.
                msg = f"particle left the field domain at step {step}: {exc}"
                if msg not in warnings:
                    warnings.append(msg)
                nlon, nlat = lon, lat
            if diff_sigma_m > 0:
                nlon, nlat = offset_by_metres(
                    (nlon, nlat),
                    float(rng.normal(0.0, diff_sigma_m)),
                    float(rng.normal(0.0, diff_sigma_m)),
                )
            moved.append((nlon, nlat))

        particles = moved
        when = when + timedelta(seconds=dt_s)
        c_lon, c_lat, spread = _cloud_stats(particles)
        steps.append({
            "step": step,
            "at": when,
            "hours_before": round(step * timestep_minutes / 60.0, 3),
            "lon": c_lon, "lat": c_lat,
            "spread_km": round(spread, 3),
        })

    elapsed = time.perf_counter() - started
    final_lon, final_lat, particle_spread = _cloud_stats(particles)

    # Reported uncertainty is the particle spread PLUS an advection error term
    # that grows with backtrack time. Quoting the spread alone would imply the
    # current field itself is exact, which it is not.
    model_error_km = MODEL_ERROR_KM_PER_HOUR * backtrack_hours
    uncertainty_km = math.sqrt(particle_spread**2 + model_error_km**2)

    if backtrack_hours > 24.0:
        warnings.append(
            f"backtracked {backtrack_hours:.0f} h - beyond ~24 h the origin "
            f"estimate is a wide blur, not a point (CLAUDE.md physical limits)"
        )

    sources = {
        "currents": getattr(currents, "source", getattr(currents, "name", "unknown")),
        "wind": getattr(wind, "source", getattr(wind, "name", "unknown")),
    }
    method = "analytical-advection"
    if "synthetic" in str(sources["currents"]):
        method = "analytical-advection-SYNTHETIC"
        warnings.append(
            "currents are SYNTHETIC - this origin is a demonstration, not a measurement"
        )

    origin = DriftOrigin(
        lon=final_lon,
        lat=final_lat,
        estimated_at=when,
        uncertainty_km=round(uncertainty_km, 2),
        n_particles=len(particles),
        backtrack_hours=float(backtrack_hours),
        method=method,
        track=[
            {
                "lon": round(s["lon"], 6),
                "lat": round(s["lat"], 6),
                "at": s["at"].isoformat(),
                "hours_before": s["hours_before"],
                "spread_km": s["spread_km"],
            }
            # Oldest first: the UI plays this forward from the origin, so the
            # animation runs the way the oil actually travelled.
            for s in reversed(steps)
        ],
    )

    log.info(
        "Backtracked %.1f h to (%.4f, %.4f) +/- %.1f km using %d particles in %.2fs",
        backtrack_hours, final_lon, final_lat, uncertainty_km, len(particles), elapsed,
    )

    return DriftResult(
        origin=origin,
        particles_lonlat=particles,
        steps=steps,
        warnings=warnings,
        stats={
            "elapsed_s": round(elapsed, 3),
            "n_steps": n_steps,
            "timestep_minutes": timestep_minutes,
            "particle_spread_km": round(particle_spread, 2),
            "model_error_km": round(model_error_km, 2),
            "sources": sources,
            "displacement_km": round(
                haversine_km((steps[0]["lon"], steps[0]["lat"]), (final_lon, final_lat)), 2
            ),
        },
    )
