"""Route a confirmed slick to the right attribution path by its shape.

    long thin streak, tapering  -> a moving vessel was discharging -> vessel path
    irregular blob, fixed       -> wreck / platform / natural seep -> infrastructure

Getting this wrong in the blob direction is the worst failure the system can
produce: accusing a passing ship of a natural seep. So the blob branch is
deliberately sticky - anything that looks fixed, or sits on a known seep or
installation, leaves the vessel path entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.contracts import Morphology, SourceType
from core.geo import haversine_km
from detect.polygonize import RegionFeatures

log = logging.getLogger(__name__)

# A vessel discharge is long and thin. Published bilge-dump studies put
# typical length:width well above 10:1; blooms and rain cells sit near 1-3:1.
LINEAR_ELONGATION_MIN = 6.0
LINEAR_COMPACTNESS_MAX = 0.35
BLOB_COMPACTNESS_MIN = 0.45

# How close a slick must sit to a catalogued seep or installation before we
# treat it as that feature rather than a vessel discharge.
INFRASTRUCTURE_MATCH_KM = 12.0


@dataclass
class KnownSource:
    """A catalogued fixed source of surface oil."""

    name: str
    lon: float
    lat: float
    kind: SourceType  # "natural_seep" | "infrastructure"
    radius_km: float = INFRASTRUCTURE_MATCH_KM
    note: str = ""


# Seeded with the well-documented persistent sites named in CLAUDE.md. This
# list is meant to be extended from a real seep/platform catalogue; every
# entry here is public knowledge.
KNOWN_SOURCES: list[KnownSource] = [
    KnownSource(
        name="Taylor Energy MC-20",
        lon=-88.970, lat=28.936, kind="infrastructure", radius_km=15.0,
        note="Continuous leak since 2004; visible on nearly every pass.",
    ),
    KnownSource(
        name="Coal Oil Point seep field",
        lon=-119.883, lat=34.391, kind="natural_seep", radius_km=12.0,
        note="One of the largest natural marine seeps on Earth.",
    ),
    KnownSource(
        name="Campeche Bay seeps",
        lon=-92.200, lat=19.700, kind="natural_seep", radius_km=40.0,
        note="Extensive natural seepage in the southern Gulf of Mexico.",
    ),
    KnownSource(
        name="MSC ELSA 3 wreck",
        lon=76.136, lat=9.3125, kind="infrastructure", radius_km=10.0,
        note="Sank 25 May 2025 off Kerala; wreck leaked for weeks.",
    ),
]


@dataclass
class MorphologyVerdict:
    morphology: Morphology
    source_type: SourceType
    reason: str
    matched_source: KnownSource | None = None
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def goes_to_vessel_attribution(self) -> bool:
        return self.source_type == "vessel"


def match_known_source(
    lon: float, lat: float, catalogue: list[KnownSource] | None = None
) -> KnownSource | None:
    """Nearest catalogued seep/installation within its own radius, if any."""
    sources = KNOWN_SOURCES if catalogue is None else catalogue
    best: tuple[float, KnownSource] | None = None
    for src in sources:
        d = haversine_km((lon, lat), (src.lon, src.lat))
        if d <= src.radius_km and (best is None or d < best[0]):
            best = (d, src)
    return best[1] if best else None


def classify_morphology(
    region: RegionFeatures,
    catalogue: list[KnownSource] | None = None,
    seen_in_previous_passes: bool = False,
) -> MorphologyVerdict:
    """Decide the attribution route for one confirmed slick.

    The known-source check runs FIRST and wins outright. A natural seep that
    happens to be smeared into a streak by the current must not be routed to
    vessel attribution just because it looks linear.
    """
    lon, lat = region.centroid_lonlat
    match = match_known_source(lon, lat, catalogue)
    if match is not None:
        d = haversine_km((lon, lat), (match.lon, match.lat))
        return MorphologyVerdict(
            morphology="blob",
            source_type=match.kind,
            reason=(
                f"{d:.1f} km from catalogued {match.kind.replace('_', ' ')} "
                f"'{match.name}' - excluded from vessel attribution. {match.note}"
            ),
            matched_source=match,
            scores={"distance_km": round(d, 2)},
        )

    if seen_in_previous_passes:
        return MorphologyVerdict(
            morphology="blob",
            source_type="infrastructure",
            reason=(
                "slick recurs at this position across satellite passes - "
                "consistent with a fixed source (wreck, platform or seep), "
                "not a passing vessel"
            ),
            scores={"elongation": region.elongation},
        )

    elongation = region.elongation
    compactness = region.compactness
    scores = {
        "elongation": round(elongation, 2),
        "compactness": round(compactness, 3),
        "major_axis_km": round(region.major_axis_km, 2),
    }

    if elongation >= LINEAR_ELONGATION_MIN and compactness <= LINEAR_COMPACTNESS_MAX:
        return MorphologyVerdict(
            morphology="linear",
            source_type="vessel",
            reason=(
                f"elongation {elongation:.1f}:1 over {region.major_axis_km:.1f} km "
                f"with compactness {compactness:.2f} - consistent with discharge "
                f"from a moving vessel"
            ),
            scores=scores,
        )

    if compactness >= BLOB_COMPACTNESS_MIN:
        return MorphologyVerdict(
            morphology="blob",
            source_type="unknown",
            reason=(
                f"compact blob (compactness {compactness:.2f}, elongation "
                f"{elongation:.1f}:1) with no catalogued source nearby - "
                f"origin unresolved, not attributed to a vessel"
            ),
            scores=scores,
        )

    return MorphologyVerdict(
        morphology="unknown",
        source_type="unknown",
        reason=(
            f"shape is ambiguous (elongation {elongation:.1f}:1, compactness "
            f"{compactness:.2f}) - between a discharge streak and a fixed blob"
        ),
        scores=scores,
    )
