"""Shared dataclasses — the interface between all modules.

FROZEN. Any change here must be flagged to the team; every module imports
from this file and nothing else crosses module boundaries.

Coordinate convention used everywhere in this repo:
    bbox = (min_lon, min_lat, max_lon, max_lat)
    point = (lon, lat)
Longitude first, always. GeoJSON agrees; Shapely agrees; several APIs do not,
so conversion happens at the API boundary and nowhere else.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

BBox = tuple[float, float, float, float]

# Wind window for SAR oil detection. Soft edges on purpose — oil is
# occasionally visible outside this band, so we grade rather than cut.
WIND_LOW_CUT_MS = 2.0     # below this, calm water is indistinguishable from oil
WIND_LOW_FULL_MS = 3.5    # full confidence starts here
WIND_HIGH_FULL_MS = 7.5   # full confidence ends here
WIND_HIGH_CUT_MS = 12.0   # above this, oil mixes into the wave field

SourceType = Literal["vessel", "infrastructure", "natural_seep", "unknown"]
Morphology = Literal["linear", "blob", "unknown"]
OrbitDirection = Literal["ASCENDING", "DESCENDING"]


def utcnow() -> datetime:
    """Timezone-aware UTC now. Never use datetime.utcnow() — it returns naive."""
    return datetime.now(timezone.utc)


def wind_window_score(speed_ms: float) -> float:
    """Graded 0-1 suitability of a wind speed for SAR oil detection.

    Trapezoid: 0 below WIND_LOW_CUT, ramps to 1 by WIND_LOW_FULL, holds to
    WIND_HIGH_FULL, decays to 0 at WIND_HIGH_CUT. Graded, not a hard cutoff,
    because the published bounds themselves disagree by several m/s.
    """
    if not math.isfinite(speed_ms) or speed_ms <= WIND_LOW_CUT_MS:
        return 0.0
    if speed_ms >= WIND_HIGH_CUT_MS:
        return 0.0
    if speed_ms < WIND_LOW_FULL_MS:
        return (speed_ms - WIND_LOW_CUT_MS) / (WIND_LOW_FULL_MS - WIND_LOW_CUT_MS)
    if speed_ms <= WIND_HIGH_FULL_MS:
        return 1.0
    return (WIND_HIGH_CUT_MS - speed_ms) / (WIND_HIGH_CUT_MS - WIND_HIGH_FULL_MS)


@dataclass
class Scene:
    """One Sentinel-1 GRD acquisition."""

    scene_id: str
    acquired_at: datetime
    bbox: BBox  # min_lon, min_lat, max_lon, max_lat
    vv_path: Path
    vh_path: Path | None
    orbit_direction: OrbitDirection

    @property
    def is_dual_pol(self) -> bool:
        return self.vh_path is not None

    @property
    def centre(self) -> tuple[float, float]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)


@dataclass
class WindContext:
    """Wind at a slick's location and acquisition time.

    The single strongest feature for rejecting look-alikes. A candidate
    without wind context is not a candidate.
    """

    speed_ms: float
    direction_deg: float
    source: str  # "ERA5" | "ASCAT" | "synthetic"
    window_score: float  # 0-1, graded; see wind_window_score()

    @classmethod
    def from_speed(
        cls,
        speed_ms: float,
        direction_deg: float = 0.0,
        source: str = "ERA5",
    ) -> WindContext:
        return cls(
            speed_ms=speed_ms,
            direction_deg=direction_deg,
            source=source,
            window_score=wind_window_score(speed_ms),
        )


@dataclass
class SlickCandidate:
    """A dark patch that survived (or failed) the look-alike physics check."""

    candidate_id: str
    scene_id: str
    polygon_wkt: str
    area_km2: float
    elongation: float       # major/minor axis ratio; high => linear discharge
    compactness: float      # 4*pi*area / perimeter^2; 1.0 == perfect circle
    damping_ratio: float    # mean backscatter inside / surrounding sea
    wind: WindContext
    p_oil: float                     # after look-alike rejection
    rejected_reason: str | None      # e.g. "wind 1.2 m/s below threshold"
    morphology: Morphology
    centroid: tuple[float, float] = (0.0, 0.0)  # lon, lat
    texture_homogeneity: float = 0.0
    texture_contrast: float = 0.0
    texture_variance: float = 0.0
    vh_vv_ratio: float | None = None  # None on single-pol scenes
    feature_contributions: dict[str, float] = field(default_factory=dict)

    @property
    def is_rejected(self) -> bool:
        return self.rejected_reason is not None


@dataclass
class DriftOrigin:
    """Where and when a slick most likely started, from backward drift."""

    lon: float
    lat: float
    estimated_at: datetime
    uncertainty_km: float
    n_particles: int
    backtrack_hours: float = 0.0
    method: str = "unknown"  # "opendrift-openoil" | "analytical-advection" | "stub"
    # Ordered oldest-first: the particle-cloud centroid at each backward step.
    # This is what the UI animates.
    track: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_reliable(self) -> bool:
        """Drift error grows with time; beyond ~24 h the origin is a wide blur."""
        return self.backtrack_hours <= 24.0 and self.uncertainty_km <= 50.0


@dataclass
class VesselCandidate:
    """One ranked suspect. Never presented as a conclusion."""

    mmsi: str
    name: str | None
    vessel_type: str | None
    flag: str | None
    parity: float        # 0-1, course alignment with slick long axis
    proximity: float     # 0-1, closeness to drift origin
    temporality: float   # 0-1, recency of passage
    score: float
    went_dark: bool      # AIS gap coincident with origin
    evidence: str        # human-readable, shown on screen
    track: list[dict[str, Any]] = field(default_factory=list)
    closest_approach_km: float = float("nan")
    closest_approach_at: datetime | None = None


@dataclass
class Attribution:
    """The end of the pipeline. Ranked candidates, or an honest abstention."""

    candidate_id: str
    origin: DriftOrigin | None
    source_type: SourceType
    candidates: list[VesselCandidate]  # ranked, may be empty
    abstained: bool
    abstain_reason: str | None
    evidence: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Serialisation. The API and the cache both need these; nothing else should
# hand-roll dataclass -> dict conversion.
# --------------------------------------------------------------------------

def _encode(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        # NaN/Inf are not valid JSON and silently become `NaN` tokens that
        # break every strict parser downstream, browsers included.
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def to_dict(obj: Any) -> dict[str, Any]:
    """Dataclass -> JSON-safe dict."""
    return _encode(asdict(obj))


def to_json(obj: Any, **kwargs: Any) -> str:
    return json.dumps(to_dict(obj), **kwargs)
