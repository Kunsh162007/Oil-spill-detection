"""Internal-wave rejection by periodicity.

A single internal-wave band looks exactly like a thin oil streak: elongated,
darker than the surrounding sea, plausible damping. Judged one at a time,
every band passes - and CLAUDE.md names internal waves as one of the two
look-alike clusters that break real systems.

What gives them away is not any single band, but the SET: internal waves
arrive as a train of several near-parallel, similarly-sized bands at a
regular spacing. Oil discharged by a vessel does not come in evenly spaced
repeats.

So this check is scene-level, and runs after per-region physics. It is the
one place where a region's verdict depends on its neighbours.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from core.geo import axial_difference_deg, haversine_km
from detect.polygonize import RegionFeatures

log = logging.getLogger(__name__)

# A train needs at least this many bands. Two parallel streaks happen by
# chance - and a vessel that turned can genuinely leave two - so requiring
# three keeps a real double-discharge out of the reject pile.
MIN_TRAIN_SIZE = 3
MAX_AXIS_SPREAD_DEG = 20.0     # bands must be near-parallel
MAX_SIZE_RATIO = 3.0           # and of comparable size
MAX_SPACING_CV = 0.45          # spacing must be regular, not scattered


@dataclass
class WaveTrain:
    """A set of regions that together look like an internal-wave packet."""

    labels: list[int]
    mean_axis_deg: float
    mean_spacing_km: float
    spacing_cv: float
    members: list[RegionFeatures] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.labels)

    def reason(self) -> str:
        return (
            f"one of {self.size} near-parallel bands (axis ~{self.mean_axis_deg:.0f} deg) "
            f"evenly spaced every {self.mean_spacing_km:.1f} km "
            f"(spacing variation {self.spacing_cv:.0%}) - the signature of an "
            f"internal-wave train, not a vessel discharge"
        )


def _perpendicular_offsets(
    members: list[RegionFeatures], axis_deg: float
) -> list[float]:
    """Distance of each band from a reference line, measured across the axis.

    Projecting the centroids onto the direction PERPENDICULAR to the shared
    axis turns a 2-D scatter of bands into a 1-D sequence, which is what
    makes "evenly spaced" a testable statement.
    """
    ref = members[0].centroid_lonlat
    perp = math.radians((axis_deg + 90.0) % 360.0)
    offsets = []
    for m in members:
        lon, lat = m.centroid_lonlat
        d = haversine_km(ref, (lon, lat))
        if d == 0:
            offsets.append(0.0)
            continue
        from core.geo import bearing_deg

        brg = math.radians(bearing_deg(ref, (lon, lat)))
        # Component of the separation along the cross-axis direction.
        offsets.append(d * math.cos(brg - perp))
    return sorted(offsets)


def _spacing_stats(offsets: list[float]) -> tuple[float, float]:
    """Mean spacing between consecutive bands and its coefficient of variation."""
    if len(offsets) < 2:
        return 0.0, float("inf")
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    gaps = [g for g in gaps if g > 1e-6]
    if not gaps:
        return 0.0, float("inf")
    mean = float(np.mean(gaps))
    if mean <= 0:
        return 0.0, float("inf")
    return mean, float(np.std(gaps) / mean)


def find_wave_trains(
    regions: list[RegionFeatures],
    min_train_size: int = MIN_TRAIN_SIZE,
    max_axis_spread_deg: float = MAX_AXIS_SPREAD_DEG,
    max_size_ratio: float = MAX_SIZE_RATIO,
    max_spacing_cv: float = MAX_SPACING_CV,
) -> list[WaveTrain]:
    """Group elongated regions into internal-wave trains, if any exist."""
    # Only elongated features are candidates; a bloom is never a wave band.
    candidates = [r for r in regions if r.elongation >= 3.0 and r.area_km2 > 0]
    if len(candidates) < min_train_size:
        return []

    used: set[int] = set()
    trains: list[WaveTrain] = []

    # Greedy: seed on each unused region, gather everything parallel and
    # comparably sized, then test whether the spacing is regular.
    for seed in sorted(candidates, key=lambda r: -r.area_km2):
        if seed.label in used:
            continue
        group = [seed]
        for other in candidates:
            if other.label == seed.label or other.label in used:
                continue
            if axial_difference_deg(other.orientation_deg, seed.orientation_deg) > max_axis_spread_deg:
                continue
            ratio = max(other.area_km2, seed.area_km2) / max(min(other.area_km2, seed.area_km2), 1e-9)
            if ratio > max_size_ratio:
                continue
            group.append(other)

        if len(group) < min_train_size:
            continue

        axes = [m.orientation_deg for m in group]
        mean_axis = _mean_axis(axes)
        offsets = _perpendicular_offsets(group, mean_axis)
        spacing, cv = _spacing_stats(offsets)
        if not math.isfinite(cv) or cv > max_spacing_cv or spacing <= 0:
            continue

        labels = [m.label for m in group]
        used.update(labels)
        trains.append(WaveTrain(
            labels=labels,
            mean_axis_deg=mean_axis,
            mean_spacing_km=round(spacing, 3),
            spacing_cv=round(cv, 3),
            members=group,
        ))
        log.info(
            "Internal-wave train: %d bands, axis %.0f deg, spacing %.1f km (cv %.2f)",
            len(group), mean_axis, spacing, cv,
        )

    return trains


def _mean_axis(axes: list[float]) -> float:
    """Circular mean of undirected axes (period 180, not 360)."""
    x = sum(math.sin(math.radians(2 * a)) for a in axes)
    y = sum(math.cos(math.radians(2 * a)) for a in axes)
    return (math.degrees(math.atan2(x, y)) / 2.0) % 180.0


def wave_train_rejections(regions: list[RegionFeatures], **kwargs) -> dict[int, str]:
    """Map region label -> rejection reason for every band in a wave train."""
    rejections: dict[int, str] = {}
    for train in find_wave_trains(regions, **kwargs):
        reason = train.reason()
        for label in train.labels:
            rejections[label] = reason
    return rejections
