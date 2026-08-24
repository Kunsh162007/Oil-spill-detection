"""Rank vessels against a slick origin. Correlation, never proof.

Three measurements, each 0-1, combined by configured weights:

  parity      how parallel the vessel's course is to the slick's long axis
  proximity   how near it passed to the drift origin
  temporality how recently it passed, relative to the estimated release

Plus a bonus when the vessel's AIS went dark at the origin - a ship going
silent exactly where a slick starts is a stronger signal than one that
stayed visible.

The output is a ranked list with reasons attached. Every surface that shows
it must say that a correlation is a lead for investigators, not evidence.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime

from core.contracts import DriftOrigin, VesselCandidate
from core.geo import axial_difference_deg, haversine_km
from attribute.ais import AISGap, VesselTrack, find_gaps

log = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"parity": 0.3, "proximity": 0.4, "temporality": 0.3}


@dataclass
class ScoringContext:
    """Everything a candidate is scored against."""

    origin: DriftOrigin
    slick_axis_deg: float | None       # slick long axis, 0-180, undirected
    release_time: datetime
    search_radius_km: float = 50.0
    window_hours: float = 8.0
    weights: dict[str, float] | None = None
    dark_vessel_bonus: float = 0.15
    gap_min_minutes: float = 30.0


def parity_score(track: VesselTrack, slick_axis_deg: float | None) -> tuple[float, str]:
    """How closely the vessel's course lines up with the slick's long axis.

    Compared as undirected AXES, not bearings: a slick has no head or tail,
    so a ship steaming 090 and one steaming 270 are equally parallel to an
    east-west streak. Using a bearing difference here would score the
    reciprocal course as maximally misaligned - a 90 degree error.
    """
    if slick_axis_deg is None:
        return 0.0, "no slick axis available (blob morphology) - parity not assessable"

    course = track.mean_course_deg()
    if course is None:
        return 0.0, "vessel course unknown - too few AIS positions to derive a heading"

    diff = axial_difference_deg(course, slick_axis_deg)  # 0-90
    score = max(0.0, 1.0 - diff / 90.0)
    return score, (
        f"course {course:.0f} deg vs slick axis {slick_axis_deg:.0f} deg "
        f"({diff:.0f} deg off parallel)"
    )


def proximity_score(
    track: VesselTrack, origin: DriftOrigin, search_radius_km: float
) -> tuple[float, str, float, datetime | None]:
    """How near the vessel passed to the estimated origin.

    The tolerance is the drift uncertainty itself, not a fixed number: when
    the origin is a 20 km blur, passing 15 km away is a close approach, and
    when it is tight, it is not.
    """
    closest = track.closest_approach((origin.lon, origin.lat))
    if closest is None:
        return 0.0, "no AIS positions for this vessel in the window", float("nan"), None

    distance_km, position = closest
    tolerance = max(origin.uncertainty_km, 1.0)

    if distance_km > search_radius_km:
        return 0.0, (
            f"closest approach {distance_km:.1f} km, beyond the "
            f"{search_radius_km:.0f} km search radius"
        ), distance_km, position.timestamp

    # Gaussian falloff on the drift uncertainty scale.
    score = math.exp(-0.5 * (distance_km / tolerance) ** 2)
    return score, (
        f"passed {distance_km:.1f} km from the estimated origin "
        f"(drift uncertainty +/-{origin.uncertainty_km:.1f} km)"
    ), distance_km, position.timestamp


def temporality_score(
    approach_time: datetime | None, release_time: datetime, window_hours: float
) -> tuple[float, str]:
    """How close in time the vessel's passage was to the estimated release.

    Asymmetric on purpose. A ship that passed BEFORE the estimated release
    could have done it; one that arrived well AFTER the oil was already in
    the water could not, so late arrivals decay roughly three times faster.
    """
    if approach_time is None:
        return 0.0, "no timestamp for the closest approach"

    delta_h = (approach_time - release_time).total_seconds() / 3600.0
    scale = max(window_hours, 0.5)

    if delta_h <= 0:
        score = math.exp(-abs(delta_h) / scale)
        when = f"{abs(delta_h):.1f} h before the estimated release"
    else:
        score = math.exp(-3.0 * delta_h / scale)
        when = f"{delta_h:.1f} h AFTER the estimated release"

    return score, f"closest approach {when}"


def dark_vessel_check(
    track: VesselTrack, ctx: ScoringContext
) -> tuple[bool, AISGap | None, str]:
    """Did this vessel go silent near the origin, around the release time?"""
    gaps = find_gaps(track, min_gap_minutes=ctx.gap_min_minutes)
    if not gaps:
        return False, None, ""

    tolerance_km = max(ctx.origin.uncertainty_km * 2.0, 10.0)
    for gap in gaps:
        d = haversine_km((ctx.origin.lon, ctx.origin.lat), gap.midpoint)
        if d > tolerance_km:
            continue
        # The release must fall inside the silence, with a little slack.
        pad = 1.0  # hours
        from datetime import timedelta

        if not (gap.gap_start - timedelta(hours=pad) <= ctx.release_time
                <= gap.gap_end + timedelta(hours=pad)):
            continue
        return True, gap, (
            f"AIS silent for {gap.duration_hours:.1f} h ({gap.gap_start:%d %b %H:%M} - "
            f"{gap.gap_end:%d %b %H:%M} UTC), {d:.1f} km from the estimated origin, "
            f"spanning the estimated release time"
        )
    return False, None, ""


# Necessary conditions. A vessel failing either of these could not physically
# be the source, no matter how well the other terms score. Without these gates
# a plain weighted sum lets strong parity carry a ship that was 120 km away,
# or one that arrived hours after the oil was already on the water.
MIN_PROXIMITY = 1e-6      # zero means "never entered the search radius"
MIN_TEMPORALITY = 0.05    # arrived so late the slick already existed


def disqualify(candidate: VesselCandidate) -> str | None:
    """Reason this vessel cannot be the source, or None if it stays in play."""
    if candidate.proximity <= MIN_PROXIMITY:
        return "never came within the search radius of the estimated origin"
    if candidate.temporality < MIN_TEMPORALITY:
        return "passed only after the slick was already on the water"
    return None


def score_vessel(track: VesselTrack, ctx: ScoringContext) -> VesselCandidate:
    """Score one vessel against the slick origin."""
    weights = ctx.weights or DEFAULT_WEIGHTS

    parity, parity_note = parity_score(track, ctx.slick_axis_deg)
    proximity, prox_note, distance_km, approach_time = proximity_score(
        track, ctx.origin, ctx.search_radius_km
    )
    temporality, time_note = temporality_score(
        approach_time, ctx.release_time, ctx.window_hours
    )

    base = (
        weights.get("parity", 0.3) * parity
        + weights.get("proximity", 0.4) * proximity
        + weights.get("temporality", 0.3) * temporality
    )

    went_dark, gap, dark_note = dark_vessel_check(track, ctx)
    score = min(1.0, base + (ctx.dark_vessel_bonus if went_dark else 0.0))

    parts = [prox_note, time_note, parity_note]
    if went_dark:
        parts.append("WENT DARK: " + dark_note)
    evidence = "; ".join(p for p in parts if p)

    return VesselCandidate(
        mmsi=track.mmsi,
        name=track.name,
        vessel_type=track.vessel_type,
        flag=track.flag,
        parity=round(parity, 4),
        proximity=round(proximity, 4),
        temporality=round(temporality, 4),
        score=round(score, 4),
        went_dark=went_dark,
        evidence=evidence,
        track=track.to_geojson_coords(),
        closest_approach_km=round(distance_km, 3) if math.isfinite(distance_km) else float("nan"),
        closest_approach_at=approach_time,
    )


def rank_vessels(
    tracks: list[VesselTrack], ctx: ScoringContext, top_n: int = 3
) -> list[VesselCandidate]:
    """Score every vessel and return the best few, highest first.

    Vessels scoring zero are dropped rather than padded into the list: a
    ranked list of three where two are meaningless invites the reader to
    treat them as suspects.
    """
    scored: list[VesselCandidate] = []
    rejected: list[tuple[str, str]] = []

    for t in tracks:
        candidate = score_vessel(t, ctx)
        reason = disqualify(candidate)
        if reason is not None:
            rejected.append((candidate.name or candidate.mmsi, reason))
            continue
        if candidate.score <= 0.0:
            continue
        scored.append(candidate)

    scored.sort(key=lambda c: c.score, reverse=True)
    for name, reason in rejected:
        log.info("Excluded %s: %s", name, reason)
    log.info(
        "Ranked %d/%d vessels (%d excluded on necessary conditions)",
        len(scored), len(tracks), len(rejected),
    )
    return scored[:top_n]
