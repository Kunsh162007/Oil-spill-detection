"""Drift backend selection: OpenOil when available, analytical otherwise.

OpenDrift's OpenOil is the reference implementation and is what we cite. It
is conda-first and frequently absent on a laptop, so the analytical RK4
integrator in drift/backward.py is a first-class fallback rather than a toy -
it uses the same fields and the same seeding, and reports which backend ran.

Whichever runs, weathering is OFF. Evaporation and emulsification are not
reversible, so running them backwards produces numbers with no physical
meaning.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

from core.contracts import DriftOrigin, SlickCandidate
from drift.backward import DriftResult, backtrack
from drift.readers import ConstantField, SyntheticField, VectorField, WindField, zero_field

log = logging.getLogger(__name__)


def opendrift_available() -> bool:
    try:
        import opendrift  # noqa: F401
        return True
    except ImportError:
        return False


def _parse_polygon(wkt: str) -> list[tuple[float, float]]:
    """WKT POLYGON -> lon/lat vertex list."""
    from shapely import wkt as shapely_wkt

    geom = shapely_wkt.loads(wkt)
    if geom.is_empty:
        raise ValueError("Slick polygon is empty - cannot run drift")
    if geom.geom_type == "Polygon":
        return [(float(x), float(y)) for x, y in geom.exterior.coords]
    if geom.geom_type == "MultiPolygon":
        biggest = max(geom.geoms, key=lambda g: g.area)
        return [(float(x), float(y)) for x, y in biggest.exterior.coords]
    raise ValueError(f"Expected a polygon, got {geom.geom_type}")


def run_backward_drift(
    candidate: SlickCandidate,
    observed_at: datetime,
    currents: VectorField,
    wind: VectorField | None,
    backtrack_hours: float = 12.0,
    timestep_minutes: float = 30.0,
    n_particles: int = 500,
    diffusion_m2_s: float = 5.0,
    backend: str = "auto",
    seed: int = 0,
) -> DriftResult:
    """Backtrack one slick candidate to its probable origin."""
    polygon = _parse_polygon(candidate.polygon_wkt)

    if backend in ("auto", "opendrift") and opendrift_available():
        try:
            return _run_opendrift(
                polygon, observed_at, currents, wind, backtrack_hours,
                timestep_minutes, n_particles, diffusion_m2_s,
            )
        except Exception as exc:
            if backend == "opendrift":
                raise
            log.warning("OpenOil run failed (%s); using the analytical integrator", exc)
    elif backend == "opendrift":
        raise RuntimeError(
            "backend='opendrift' requested but OpenDrift is not installed. "
            "pip install opendrift, or set drift.backend to 'auto'."
        )

    return backtrack(
        polygon_lonlat=polygon,
        observed_at=observed_at,
        currents=currents,
        wind=wind,
        backtrack_hours=backtrack_hours,
        timestep_minutes=timestep_minutes,
        n_particles=n_particles,
        diffusion_m2_s=diffusion_m2_s,
        seed=seed,
    )


def _run_opendrift(
    polygon: Sequence[tuple[float, float]],
    observed_at: datetime,
    currents: VectorField,
    wind: VectorField | None,
    backtrack_hours: float,
    timestep_minutes: float,
    n_particles: int,
    diffusion_m2_s: float,
) -> DriftResult:
    """Backward OpenOil run. Weathering disabled - see module docstring."""
    import numpy as np
    from opendrift.models.openoil import OpenOil

    from core.geo import haversine_km

    model = OpenOil(loglevel=50, weathering_model=None)

    # Turn off every irreversible process. Running these backwards is not
    # physics; leaving them on silently corrupts the origin estimate.
    for key, value in {
        "processes:evaporation": False,
        "processes:emulsification": False,
        "processes:dispersion": False,
        "processes:biodegradation": False,
        "drift:vertical_mixing": False,
        "drift:horizontal_diffusivity": diffusion_m2_s,
    }.items():
        try:
            model.set_config(key, value)
        except Exception:
            log.debug("OpenOil config %s not accepted by this version", key)

    readers = _build_opendrift_readers(currents, wind)
    if readers:
        model.add_reader(readers)
    else:
        # Constant fallback fields, so a missing reader cannot silently become
        # a still ocean that reports a confident origin at the observation.
        sample = currents.sample(polygon[0][0], polygon[0][1], observed_at)
        model.fallback_values["x_sea_water_velocity"] = sample.u
        model.fallback_values["y_sea_water_velocity"] = sample.v
        if wind is not None:
            w = wind.sample(polygon[0][0], polygon[0][1], observed_at)
            model.fallback_values["x_wind"] = w.u / 0.03
            model.fallback_values["y_wind"] = w.v / 0.03

    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    model.seed_within_polygon(
        lons=lons, lats=lats, number=n_particles, time=observed_at,
    )

    model.run(
        duration=__import__("datetime").timedelta(hours=backtrack_hours),
        time_step=-int(timestep_minutes * 60),         # negative == backwards
        time_step_output=int(timestep_minutes * 60),
    )

    lon_hist = np.asarray(model.history["lon"])
    lat_hist = np.asarray(model.history["lat"])
    final_lon = float(np.ma.median(lon_hist[:, -1]))
    final_lat = float(np.ma.median(lat_hist[:, -1]))

    particles = [
        (float(lo), float(la))
        for lo, la in zip(lon_hist[:, -1], lat_hist[:, -1])
        if np.isfinite(lo) and np.isfinite(la)
    ]
    dists = [haversine_km((final_lon, final_lat), p) for p in particles]
    spread = float(np.percentile(dists, 68)) if dists else 0.0

    from drift.backward import MODEL_ERROR_KM_PER_HOUR

    model_error = MODEL_ERROR_KM_PER_HOUR * backtrack_hours
    uncertainty = float(np.hypot(spread, model_error))

    n_out = lon_hist.shape[1]
    track: list[dict[str, Any]] = []
    for step in range(n_out - 1, -1, -1):  # oldest first
        hours_before = (n_out - 1 - step) * timestep_minutes / 60.0
        track.append({
            "lon": round(float(np.ma.median(lon_hist[:, step])), 6),
            "lat": round(float(np.ma.median(lat_hist[:, step])), 6),
            "at": (observed_at - __import__("datetime").timedelta(hours=hours_before)).isoformat(),
            "hours_before": round(hours_before, 3),
            "spread_km": 0.0,
        })
    track.reverse()

    origin = DriftOrigin(
        lon=final_lon,
        lat=final_lat,
        estimated_at=observed_at - __import__("datetime").timedelta(hours=backtrack_hours),
        uncertainty_km=round(uncertainty, 2),
        n_particles=len(particles),
        backtrack_hours=float(backtrack_hours),
        method="opendrift-openoil",
        track=track,
    )
    return DriftResult(
        origin=origin,
        particles_lonlat=particles,
        steps=[],
        warnings=["weathering disabled: irreversible processes cannot be run backwards"],
        stats={"backend": "opendrift-openoil", "particle_spread_km": round(spread, 2)},
    )


def _build_opendrift_readers(currents: VectorField, wind: VectorField | None) -> list:
    """Wrap NetCDF-backed fields as OpenDrift readers where possible."""
    readers = []
    for fld in (currents, wind):
        path = getattr(fld, "path", None)
        if path is None:
            continue
        try:
            from opendrift.readers import reader_netCDF_CF_generic

            readers.append(reader_netCDF_CF_generic.Reader(str(path)))
        except Exception as exc:
            log.warning("Could not build an OpenDrift reader for %s: %s", path, exc)
    return readers


def build_fields(config, wind_speed_ms: float | None = None,
                 wind_direction_deg: float = 0.0,
                 scene_id: str | None = None) -> tuple[VectorField, VectorField]:
    """Assemble the current and wind fields named in config.

    Synthetic currents are only ever used when config asks for them by name,
    per CLAUDE.md rule 6 (never invent data).
    """
    d = config.section("drift")
    source = str(d.get("currents_source", "synthetic")).lower()

    if source == "cmems":
        from drift.readers import NetCDFField

        from core.config import resolve_path

        # A config can name one file, or a directory holding one field per
        # scene. configs/live.yaml is shared by scenes from the Mediterranean to
        # the South China Sea, and no single current field covers that, so the
        # per-scene form is what makes real currents usable there at all.
        path = d.get("currents_path")
        directory = d.get("currents_dir")
        if not path and directory and scene_id:
            candidate = resolve_path(directory) / f"{scene_id}.nc"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"drift.currents_dir is set but {candidate.name} is not in "
                    f"{candidate.parent}. Fetch it with scripts/fetch_currents.py, "
                    f"or change drift.currents_source."
                )
            path = candidate
        if not path:
            raise ValueError(
                "drift.currents_source is 'cmems' but neither drift.currents_path "
                "nor drift.currents_dir (with a scene id) resolved to a file"
            )
        currents: VectorField = NetCDFField(resolve_path(path), name="cmems")
    elif source == "constant":
        currents = ConstantField.from_speed_direction(
            float(d.get("current_speed_ms", 0.2)),
            float(d.get("current_direction_deg", 90.0)),
            source="measured-constant",
        )
    elif source == "synthetic":
        log.warning(
            "Using SYNTHETIC currents - drift origins will be tagged as a "
            "demonstration, not a measurement."
        )
        currents = SyntheticField()
    else:
        raise ValueError(f"Unknown drift.currents_source: {source!r}")

    if wind_speed_ms is None:
        wind: VectorField = zero_field("no-wind")
    else:
        wind = WindField(
            ConstantField.from_speed_direction(wind_speed_ms, wind_direction_deg, source="ERA5"),
            drift_factor=float(d.get("wind_drift_factor", 0.03)),
        )
    return currents, wind
