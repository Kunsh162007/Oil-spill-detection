"""AIS vessel tracks.

Three sources, all free:
  * Global Fishing Watch - Indian waters, ~72 h delayed, needs a token.
  * Danish Maritime Authority - open HTTP CSV, no auth, for development.
  * A local CSV cache, so a demo never depends on a live network call.

The delay is not a bug to hide. Free AIS is roughly 72 h behind, which is why
CLAUDE.md forbids the phrase "real-time" anywhere in this project. Every
track carries the age of its data so the UI can state it.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from core.geo import bearing_deg, haversine_km

log = logging.getLogger(__name__)

# A gap longer than this in a vessel's AIS record is a candidate "went dark"
# event rather than ordinary reporting jitter.
DEFAULT_GAP_MINUTES = 30.0


@dataclass
class AISPosition:
    mmsi: str
    timestamp: datetime
    lon: float
    lat: float
    sog: float | None = None      # speed over ground, knots
    cog: float | None = None      # course over ground, degrees
    name: str | None = None
    vessel_type: str | None = None
    flag: str | None = None


@dataclass
class VesselTrack:
    """One vessel's positions within the query window, time-ordered."""

    mmsi: str
    positions: list[AISPosition]
    name: str | None = None
    vessel_type: str | None = None
    flag: str | None = None
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.positions.sort(key=lambda p: p.timestamp)
        if self.name is None:
            self.name = next((p.name for p in self.positions if p.name), None)
        if self.vessel_type is None:
            self.vessel_type = next((p.vessel_type for p in self.positions if p.vessel_type), None)
        if self.flag is None:
            self.flag = next((p.flag for p in self.positions if p.flag), None)

    @property
    def start(self) -> datetime | None:
        return self.positions[0].timestamp if self.positions else None

    @property
    def end(self) -> datetime | None:
        return self.positions[-1].timestamp if self.positions else None

    def closest_approach(self, point: tuple[float, float]) -> tuple[float, AISPosition] | None:
        """Nearest position to a point, as (distance_km, position)."""
        if not self.positions:
            return None
        best = min(self.positions, key=lambda p: haversine_km(point, (p.lon, p.lat)))
        return haversine_km(point, (best.lon, best.lat)), best

    def mean_course_deg(self) -> float | None:
        """Average heading over the track, from reported COG or from geometry.

        Averaged as unit vectors: a naive mean of 350 and 010 gives 180, which
        is the exact opposite of the real answer.
        """
        courses = [p.cog for p in self.positions if p.cog is not None]
        if not courses and len(self.positions) >= 2:
            courses = [
                bearing_deg((a.lon, a.lat), (b.lon, b.lat))
                for a, b in zip(self.positions, self.positions[1:])
                if haversine_km((a.lon, a.lat), (b.lon, b.lat)) > 0.01
            ]
        if not courses:
            return None
        x = sum(math.sin(math.radians(c)) for c in courses)
        y = sum(math.cos(math.radians(c)) for c in courses)
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            return None
        return math.degrees(math.atan2(x, y)) % 360.0

    def to_geojson_coords(self) -> list[dict[str, Any]]:
        return [
            {
                "lon": round(p.lon, 6),
                "lat": round(p.lat, 6),
                "at": p.timestamp.isoformat(),
                "sog": p.sog,
                "cog": p.cog,
            }
            for p in self.positions
        ]


@dataclass
class AISGap:
    """A silence in one vessel's AIS record - a possible transponder switch-off."""

    mmsi: str
    gap_start: datetime
    gap_end: datetime
    last_lon: float
    last_lat: float
    resume_lon: float
    resume_lat: float
    name: str | None = None

    @property
    def duration_hours(self) -> float:
        return (self.gap_end - self.gap_start).total_seconds() / 3600.0

    @property
    def implied_speed_knots(self) -> float:
        """Speed needed to cover the gap. Absurd values mean bad data, not a dump."""
        d_km = haversine_km((self.last_lon, self.last_lat), (self.resume_lon, self.resume_lat))
        hours = max(self.duration_hours, 1e-6)
        return (d_km / 1.852) / hours

    @property
    def midpoint(self) -> tuple[float, float]:
        return (
            (self.last_lon + self.resume_lon) / 2.0,
            (self.last_lat + self.resume_lat) / 2.0,
        )


def find_gaps(track: VesselTrack, min_gap_minutes: float = DEFAULT_GAP_MINUTES) -> list[AISGap]:
    """Find silences in a track that are long enough to be deliberate.

    A vessel going dark exactly where a slick starts is a stronger signal than
    one that stayed visible - but only if the gap is real. Gaps implying an
    impossible speed on resume are dropped as data artefacts.
    """
    gaps: list[AISGap] = []
    threshold = timedelta(minutes=min_gap_minutes)

    for prev, nxt in zip(track.positions, track.positions[1:]):
        delta = nxt.timestamp - prev.timestamp
        if delta < threshold:
            continue
        gap = AISGap(
            mmsi=track.mmsi,
            gap_start=prev.timestamp,
            gap_end=nxt.timestamp,
            last_lon=prev.lon, last_lat=prev.lat,
            resume_lon=nxt.lon, resume_lat=nxt.lat,
            name=track.name,
        )
        # No merchant vessel sustains 60 knots; such a "gap" is two different
        # ships sharing an MMSI, or a decoding error.
        if gap.implied_speed_knots > 60.0:
            log.debug(
                "Discarding implausible gap for %s: %.0f knots implied",
                track.mmsi, gap.implied_speed_knots,
            )
            continue
        gaps.append(gap)
    return gaps


def tracks_from_positions(
    positions: Iterable[AISPosition], source: str = "unknown"
) -> list[VesselTrack]:
    """Group flat AIS positions into per-vessel tracks."""
    by_mmsi: dict[str, list[AISPosition]] = {}
    for p in positions:
        by_mmsi.setdefault(p.mmsi, []).append(p)
    return [VesselTrack(mmsi=m, positions=ps, source=source) for m, ps in by_mmsi.items()]


def _parse_timestamp(value: str) -> datetime | None:
    """Parse the several timestamp shapes these feeds use. UTC always."""
    value = (value or "").strip()
    if not value:
        return None
    for fmt in (
        "%d/%m/%Y %H:%M:%S",   # Danish Maritime Authority
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _first(row: dict[str, str], *keys: str) -> str | None:
    """First non-empty value among differently-named columns across feeds."""
    for key in keys:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return None


def load_ais_csv(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    source: str = "csv",
) -> list[VesselTrack]:
    """Load AIS from a CSV, filtered to a bbox and time window.

    Handles the Danish Maritime Authority column names and the common
    lowercase variants without needing a per-file adapter.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"AIS file not found: {p}")

    positions: list[AISPosition] = []
    destinations: dict[str, str] = {}
    skipped = 0

    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mmsi = _first(row, "MMSI", "mmsi")
            lat_s = _first(row, "Latitude", "lat", "latitude", "LAT")
            lon_s = _first(row, "Longitude", "lon", "longitude", "LON")
            ts_s = _first(row, "# Timestamp", "Timestamp", "timestamp", "BaseDateTime", "time")
            if not (mmsi and lat_s and lon_s and ts_s):
                skipped += 1
                continue
            try:
                lat, lon = float(lat_s), float(lon_s)
            except ValueError:
                skipped += 1
                continue
            # AIS carries 91/181 as "position unavailable"; those are not
            # coordinates and must never reach a distance calculation.
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                skipped += 1
                continue
            ts = _parse_timestamp(ts_s)
            if ts is None:
                skipped += 1
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            if bbox and not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue

            sog = _first(row, "SOG", "sog", "speed")
            cog = _first(row, "COG", "cog", "course")
            destination = _first(row, "Destination", "destination", "DEST")
            if destination:
                destinations.setdefault(mmsi, destination)
            positions.append(AISPosition(
                mmsi=mmsi,
                timestamp=ts,
                lon=lon, lat=lat,
                sog=float(sog) if sog and sog.replace(".", "", 1).replace("-", "", 1).isdigit() else None,
                cog=float(cog) if cog and cog.replace(".", "", 1).replace("-", "", 1).isdigit() else None,
                name=_first(row, "Name", "name", "VesselName", "shipname"),
                vessel_type=_first(row, "Ship type", "ShipType", "vessel_type", "shiptype"),
                flag=_first(row, "Flag", "flag", "country"),
            ))

    tracks = tracks_from_positions(positions, source=source)
    for track in tracks:
        # What the vessel itself broadcast, which is not the same claim as
        # where its course happens to point.
        if track.mmsi in destinations:
            track.meta["destination"] = destinations[track.mmsi]
    log.info(
        "Loaded %d AIS positions -> %d vessels from %s (%d rows skipped)",
        len(positions), len(tracks), p.name, skipped,
    )
    if not positions:
        raise ValueError(
            f"No AIS positions matched in {p.name}. Check the bbox "
            f"(lon/lat order!) and time window - these feeds return empty "
            f"rather than erroring."
        )
    return tracks


class GFWClient:
    """Global Fishing Watch API - AIS for Indian waters, ~72 h delayed.

    Free non-commercial token. Empty results are raised, not returned: GFW
    answers a slightly-wrong bbox with an empty list rather than an error,
    and a silently empty AIS query looks exactly like "no ships were there".
    """

    BASE_URL = "https://gateway.api.globalfishingwatch.org/v3"

    def __init__(self, token: str | None = None, timeout: float = 30.0) -> None:
        import os

        self.token = token or os.environ.get("GFW_TOKEN", "")
        if not self.token:
            raise ValueError(
                "No GFW token. Set GFW_TOKEN in .env (free, non-commercial) "
                "or use a local AIS CSV instead."
            )
        self.timeout = timeout

    def fetch_tracks(
        self,
        bbox: tuple[float, float, float, float],
        start: datetime,
        end: datetime,
        limit: int = 100,
    ) -> list[VesselTrack]:
        import requests

        min_lon, min_lat, max_lon, max_lat = bbox
        params = {
            "datasets[0]": "public-global-vessel-identity:latest",
            "start-date": start.date().isoformat(),
            "end-date": end.date().isoformat(),
            "limit": limit,
            # GFW expects a GeoJSON bbox; lon first, matching our convention.
            "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        }
        resp = requests.get(
            f"{self.BASE_URL}/vessels/search",
            params=params,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        entries = payload.get("entries", [])
        if not entries:
            raise ValueError(
                f"GFW returned no vessels for bbox {bbox} between {start} and {end}. "
                f"Verify lon/lat ordering and that AIS coverage exists there "
                f"(free feed lags ~72 h)."
            )
        tracks: list[VesselTrack] = []
        for entry in entries:
            mmsi = str(entry.get("mmsi") or entry.get("ssvid") or "").strip()
            if not mmsi:
                continue
            tracks.append(VesselTrack(
                mmsi=mmsi,
                positions=[],
                name=entry.get("shipname"),
                vessel_type=entry.get("vesselType"),
                flag=entry.get("flag"),
                source="gfw",
                meta={"raw": entry},
            ))
        log.info("GFW returned %d vessels for bbox %s", len(tracks), bbox)
        return tracks


def danish_ais_url(day: datetime) -> str:
    """URL for one day of Danish Maritime Authority AIS (open, no auth)."""
    return f"http://web.ais.dk/aisdata/aisdk-{day.strftime('%Y-%m-%d')}.zip"


def ais_data_age_hours(tracks: list[VesselTrack], now: datetime | None = None) -> float | None:
    """How stale the newest AIS position is. Shown in the UI, never hidden."""
    now = now or datetime.now(timezone.utc)
    latest = [t.end for t in tracks if t.end is not None]
    if not latest:
        return None
    return (now - max(latest)).total_seconds() / 3600.0
