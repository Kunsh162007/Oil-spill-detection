"""Voyage reconstruction - where a vessel came from and where it went.

The AIS window we query is only a few hours wide, so the first and last
positions in it are the start and end of the OBSERVED SEGMENT, not of the
whole voyage. Those are different claims and the UI must not conflate them:
a ship seen crossing our box was somewhere before, and kept going after.

What can be said honestly:
  * where the track we hold begins and ends, and when
  * the nearest port to each of those points
  * the destination the vessel itself broadcast, when AIS carries one
  * the course it was steering, extrapolated ahead

Anything beyond that would be invention.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.geo import bearing_deg, destination_point, haversine_km
from attribute.ais import VesselTrack

log = logging.getLogger(__name__)

# Major ports around the Indian Ocean and the routes that matter for our
# demo region. Used only to name the nearest landfall to a track endpoint.
PORTS: list[tuple[str, str, float, float]] = [
    ("Kochi", "IN", 76.267, 9.966),
    ("Vizhinjam", "IN", 76.983, 8.378),
    ("Thiruvananthapuram", "IN", 76.950, 8.487),
    ("Kollam", "IN", 76.580, 8.893),
    ("Beypore", "IN", 75.803, 11.170),
    ("New Mangalore", "IN", 74.802, 12.922),
    ("Mormugao", "IN", 73.803, 15.408),
    ("Jawaharlal Nehru (Nhava Sheva)", "IN", 72.949, 18.949),
    ("Mumbai", "IN", 72.842, 18.944),
    ("Kandla", "IN", 70.216, 23.017),
    ("Mundra", "IN", 69.703, 22.839),
    ("Tuticorin", "IN", 78.150, 8.755),
    ("Chennai", "IN", 80.300, 13.100),
    ("Colombo", "LK", 79.842, 6.951),
    ("Galle", "LK", 80.217, 6.033),
    ("Male", "MV", 73.509, 4.175),
    ("Salalah", "OM", 54.005, 16.940),
    ("Jebel Ali", "AE", 55.027, 25.011),
    ("Karachi", "PK", 66.975, 24.815),
    ("Chattogram", "BD", 91.804, 22.316),
    ("Port Klang", "MY", 101.392, 3.000),
    ("Singapore", "SG", 103.822, 1.264),
]


@dataclass
class Waypoint:
    lon: float
    lat: float
    at: datetime
    nearest_port: str | None = None
    nearest_port_km: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lon": round(self.lon, 6),
            "lat": round(self.lat, 6),
            "at": self.at.isoformat(),
            "nearest_port": self.nearest_port,
            "nearest_port_km": (
                round(self.nearest_port_km, 1) if self.nearest_port_km is not None else None
            ),
        }


@dataclass
class Voyage:
    """The observed segment of one vessel's passage."""

    mmsi: str
    name: str | None
    track_start: Waypoint
    track_end: Waypoint
    distance_km: float
    duration_hours: float
    mean_speed_knots: float
    course_deg: float | None
    declared_destination: str | None = None
    projected_arrival: dict[str, Any] | None = None
    coverage_note: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mmsi": self.mmsi,
            "name": self.name,
            "from": self.track_start.to_dict(),
            "to": self.track_end.to_dict(),
            "distance_km": round(self.distance_km, 2),
            "duration_hours": round(self.duration_hours, 2),
            "mean_speed_knots": round(self.mean_speed_knots, 1),
            "course_deg": round(self.course_deg, 1) if self.course_deg is not None else None,
            "declared_destination": self.declared_destination,
            "projected_arrival": self.projected_arrival,
            "coverage_note": self.coverage_note,
            **self.extras,
        }


def nearest_port(lon: float, lat: float, max_km: float = 400.0):
    """Closest catalogued port to a point, or None if nothing is near."""
    best = None
    for name, flag, plon, plat in PORTS:
        d = haversine_km((lon, lat), (plon, plat))
        if d <= max_km and (best is None or d < best[1]):
            best = (f"{name} ({flag})", d)
    return best


def project_arrival(
    end_lon: float, end_lat: float, course_deg: float | None,
    speed_knots: float, max_hours: float = 48.0,
):
    """Extrapolate the vessel's heading to the port it appears bound for.

    Explicitly a PROJECTION along the last observed course, not a prediction
    and not an AIS-declared destination. It answers "if it held this course,
    where would it make landfall" - and the payload says exactly that.
    """
    if course_deg is None or speed_knots <= 0.5:
        return None

    reach_km = min(speed_knots * 1.852 * max_hours, 2000.0)
    best = None
    for name, flag, plon, plat in PORTS:
        distance = haversine_km((end_lon, end_lat), (plon, plat))
        if distance > reach_km or distance < 5.0:
            continue
        heading = bearing_deg((end_lon, end_lat), (plon, plat))
        # How far off the current course this port lies.
        off = abs((heading - course_deg + 180.0) % 360.0 - 180.0)
        if off > 30.0:
            continue
        score = off + distance / 100.0
        if best is None or score < best[0]:
            best = (score, name, flag, distance, off, heading)

    if best is None:
        # Nothing plausible ahead: still show where the course points.
        ahead = destination_point((end_lon, end_lat), course_deg, min(reach_km, 300.0))
        return {
            "port": None,
            "lon": round(ahead[0], 6),
            "lat": round(ahead[1], 6),
            "basis": "course projection only - no catalogued port lies ahead",
            "hours": round(min(reach_km, 300.0) / (speed_knots * 1.852), 1),
        }

    _, name, flag, distance, off, _ = best
    port = next(p for p in PORTS if p[0] == name)
    return {
        "port": f"{name} ({flag})",
        "lon": port[2],
        "lat": port[3],
        "distance_km": round(distance, 1),
        "off_course_deg": round(off, 1),
        "hours": round(distance / (speed_knots * 1.852), 1),
        "basis": (
            "projected by extrapolating the last observed course - "
            "not a declared destination"
        ),
    }


def build_voyage(track: VesselTrack, declared_destination: str | None = None) -> Voyage | None:
    """Reconstruct the observed segment of one vessel's passage."""
    if len(track.positions) < 2:
        return None

    first, last = track.positions[0], track.positions[-1]
    distance = sum(
        haversine_km((a.lon, a.lat), (b.lon, b.lat))
        for a, b in zip(track.positions, track.positions[1:])
    )
    duration = (last.timestamp - first.timestamp).total_seconds() / 3600.0
    speed_knots = (distance / 1.852) / duration if duration > 1e-6 else 0.0
    if speed_knots <= 0.1:
        reported = [p.sog for p in track.positions if p.sog]
        speed_knots = float(sum(reported) / len(reported)) if reported else 0.0

    course = track.mean_course_deg()

    start_port = nearest_port(first.lon, first.lat)
    end_port = nearest_port(last.lon, last.lat)

    return Voyage(
        mmsi=track.mmsi,
        name=track.name,
        track_start=Waypoint(
            first.lon, first.lat, first.timestamp,
            start_port[0] if start_port else None,
            start_port[1] if start_port else None,
        ),
        track_end=Waypoint(
            last.lon, last.lat, last.timestamp,
            end_port[0] if end_port else None,
            end_port[1] if end_port else None,
        ),
        distance_km=distance,
        duration_hours=duration,
        mean_speed_knots=speed_knots,
        course_deg=course,
        declared_destination=declared_destination,
        projected_arrival=project_arrival(last.lon, last.lat, course, speed_knots),
        coverage_note=(
            "Start and end are the limits of the AIS window we queried, not of "
            "the vessel's whole voyage. It was under way before the first fix "
            "and continued after the last."
        ),
        extras={"n_positions": len(track.positions)},
    )
