"""Dark vessels - ships that switched off AIS near a slick origin.

Often the most interesting answer. A vessel that was broadcasting normally,
went silent exactly where a slick starts, then reappeared afterwards, is a
stronger signal than one that stayed honest the whole time.

Kept separate from attribute/scoring.py because this is a distinct query:
scoring asks "who was there?", this asks "who stopped being visible there?".
Both feed the evidence bundle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.contracts import DriftOrigin
from core.geo import haversine_km
from attribute.ais import AISGap, VesselTrack, find_gaps

log = logging.getLogger(__name__)

# Common innocent explanations for an AIS gap. Coverage holes are the big one:
# terrestrial AIS receivers thin out offshore, and satellite AIS has revisit
# gaps, so a silence far from shore is weak evidence on its own.
COASTAL_COVERAGE_KM = 60.0


@dataclass
class DarkEvent:
    """One AIS gap assessed against a slick origin."""

    gap: AISGap
    distance_km: float
    spans_release: bool
    suspicion: float          # 0-1
    reason: str
    caveats: list[str]

    @property
    def mmsi(self) -> str:
        return self.gap.mmsi


def assess_gap(
    gap: AISGap,
    origin: DriftOrigin,
    release_time: datetime,
    tolerance_km: float | None = None,
    slack_hours: float = 1.0,
) -> DarkEvent:
    """Score how suspicious one AIS gap is, with its caveats attached."""
    tolerance = tolerance_km or max(origin.uncertainty_km * 2.0, 10.0)
    distance = haversine_km((origin.lon, origin.lat), gap.midpoint)

    spans = (
        gap.gap_start - timedelta(hours=slack_hours)
        <= release_time
        <= gap.gap_end + timedelta(hours=slack_hours)
    )

    caveats: list[str] = []
    suspicion = 0.0

    if distance <= tolerance and spans:
        # Closeness and timing both matter; a long silence right at the origin
        # at exactly the release time is the strong case.
        proximity_term = max(0.0, 1.0 - distance / max(tolerance, 1e-6))
        duration_term = min(gap.duration_hours / 4.0, 1.0)
        suspicion = 0.6 * proximity_term + 0.4 * duration_term
        reason = (
            f"AIS silent {gap.duration_hours:.1f} h from {gap.gap_start:%d %b %H:%M} to "
            f"{gap.gap_end:%d %b %H:%M} UTC, {distance:.1f} km from the estimated "
            f"origin, spanning the estimated release"
        )
    elif distance <= tolerance:
        suspicion = 0.15
        reason = (
            f"AIS gap {distance:.1f} km from the origin, but it does not span "
            f"the estimated release time"
        )
    else:
        reason = f"AIS gap {distance:.1f} km away - outside the origin tolerance"

    # Honest caveats. A gap is not proof of intent.
    if gap.duration_hours < 1.0:
        caveats.append("gap is short; routine transponder or receiver dropout is common")
    if gap.implied_speed_knots > 25.0:
        caveats.append(
            f"resume position implies {gap.implied_speed_knots:.0f} knots - "
            f"likely a coverage hole rather than a switch-off"
        )
    caveats.append(
        "AIS gaps also arise from receiver coverage limits, equipment faults "
        "and satellite revisit - this is a lead, not proof"
    )

    return DarkEvent(
        gap=gap,
        distance_km=round(distance, 2),
        spans_release=spans,
        suspicion=round(min(suspicion, 1.0), 4),
        reason=reason,
        caveats=caveats,
    )


def find_dark_vessels(
    tracks: list[VesselTrack],
    origin: DriftOrigin,
    release_time: datetime,
    min_gap_minutes: float = 30.0,
    min_suspicion: float = 0.2,
) -> list[DarkEvent]:
    """Every suspicious AIS gap near the origin, most suspicious first."""
    events: list[DarkEvent] = []
    for track in tracks:
        for gap in find_gaps(track, min_gap_minutes=min_gap_minutes):
            event = assess_gap(gap, origin, release_time)
            if event.suspicion >= min_suspicion:
                events.append(event)

    events.sort(key=lambda e: e.suspicion, reverse=True)
    log.info(
        "Found %d dark-vessel events above suspicion %.2f across %d tracks",
        len(events), min_suspicion, len(tracks),
    )
    return events


def summarise(events: list[DarkEvent]) -> dict[str, Any]:
    """Compact summary for the evidence bundle."""
    return {
        "n_events": len(events),
        "events": [
            {
                "mmsi": e.mmsi,
                "name": e.gap.name,
                "distance_km": e.distance_km,
                "duration_hours": round(e.gap.duration_hours, 2),
                "spans_release": e.spans_release,
                "suspicion": e.suspicion,
                "reason": e.reason,
                "caveats": e.caveats,
                "gap_start": e.gap.gap_start.isoformat(),
                "gap_end": e.gap.gap_end.isoformat(),
                "last_seen": {"lon": e.gap.last_lon, "lat": e.gap.last_lat},
                "resumed": {"lon": e.gap.resume_lon, "lat": e.gap.resume_lat},
            }
            for e in events
        ],
    }
