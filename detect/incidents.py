"""Documented spill incidents - independent, verified ground truth.

Everything else in this repo produces *detections*: a dark patch our physics
says is probably oil. This module holds something categorically different -
spills that are known to have happened, from public incident registries.

Two uses, and they must not be confused:

  1. A GLOBAL MAP LAYER of real events. These are confirmed spills, not model
     output, so they answer "show me actual oil spills worldwide" honestly in
     a way a detector on synthetic imagery never could.

  2. CORROBORATION. A SAR detection that coincides in space and time with a
     documented incident is independently supported. That is the strongest
     confidence signal available to us, because the confirmation comes from
     outside the model entirely.

Sources are recorded per record. A detection is never promoted to "confirmed"
without naming which registry entry confirmed it.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from core.geo import haversine_km

log = logging.getLogger(__name__)

# How close in space and time a detection must be to a documented incident
# before we call it corroborated.
#
# The search radius CANNOT be a constant. Oil moves. A slick found the same day
# as an incident should be right on top of it, but one found a fortnight later
# has been carried by wind and current the whole time, and demanding it still
# be within 60 km asks it to have stayed put - which is the one thing oil never
# does. MSC ELSA 3 is exactly this case: real detections 15 days after the
# sinking sat 87 km away and were scored as uncorroborated, because the
# threshold assumed a stationary slick.
#
# So the radius grows with elapsed time at a conservative surface-drift rate.
# Wind-driven leeway alone is about 3% of wind speed - at 6 m/s that is
# 0.18 m/s, ~15 km/day - and currents add to it. 20 km/day sits below the
# 0.3 m/s (26 km/day) the ELSA3 validation quotes, so it errs toward matching
# too little rather than too much.
CORROBORATION_BASE_KM = 30.0        # registry position error + slick extent
CORROBORATION_DRIFT_KM_PER_DAY = 20.0
CORROBORATION_MAX_KM = 250.0        # past this, "nearby" means nothing
CORROBORATION_KM = CORROBORATION_BASE_KM   # back-compat for callers

CORROBORATION_DAYS_BEFORE = 2.0
CORROBORATION_DAYS_AFTER = 14.0

# A persistent source keeps replenishing, so its slick's head sits on the
# source while the tail streams away. There is no single elapsed time to scale
# by, so it gets a fixed allowance of a few days' drift rather than the full cap.
CORROBORATION_PERSISTENT_KM = 100.0


def corroboration_radius_km(days_after: float | None,
                            base_km: float = CORROBORATION_BASE_KM) -> float:
    """How far a slick could plausibly have drifted since an incident.

    days_after is None (unknown date) or negative (detection precedes the
    incident) -> the base radius, since there is no elapsed time to allow for.
    """
    if days_after is None or days_after <= 0.0:
        return base_km
    return min(base_km + CORROBORATION_DRIFT_KM_PER_DAY * days_after,
               CORROBORATION_MAX_KM)

# Commodity strings that indicate petroleum. The registries are free text, so
# this is a keyword filter rather than a controlled vocabulary.
OIL_KEYWORDS = (
    "oil", "crude", "diesel", "petrol", "gasoline", "bunker", "fuel",
    "hydraulic", "lube", "condensate", "naphtha", "kerosene", "jet a",
    "bilge", "sheen", "tar", "asphalt", "bitumen", "hfo", "ifo",
)
# Explicitly NOT petroleum, even though the word "oil" may appear.
NON_OIL_KEYWORDS = (
    "vegetable oil", "palm oil", "soybean", "canola", "corn oil", "fish oil",
    "molasses", "sulfuric", "caustic", "ammonia", "fertilizer", "coal",
    "plastic", "container", "chemical only",
)


@dataclass
class SpillIncident:
    """One documented spill from a public registry."""

    incident_id: str
    name: str
    lon: float
    lat: float
    occurred_at: datetime | None
    commodity: str | None
    source: str                      # which registry this came from
    location: str | None = None
    max_release_gallons: float | None = None
    threat: str | None = None
    description: str | None = None
    url: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_petroleum(self) -> bool:
        """Whether the spilled substance is actually oil.

        The point of the whole map is oil, and these registries also record
        chemical, vegetable-oil and container losses. Showing those as "oil
        spills" would be wrong.
        """
        text = f"{self.commodity or ''} {self.name or ''}".lower()
        if any(bad in text for bad in NON_OIL_KEYWORDS):
            return False
        return any(good in text for good in OIL_KEYWORDS)

    @property
    def volume_m3(self) -> float | None:
        if self.max_release_gallons is None or not math.isfinite(self.max_release_gallons):
            return None
        return self.max_release_gallons * 0.00378541

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "name": self.name,
            "lon": round(self.lon, 6),
            "lat": round(self.lat, 6),
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else None,
            "commodity": self.commodity,
            "location": self.location,
            "source": self.source,
            "max_release_gallons": self.max_release_gallons,
            "volume_m3": round(self.volume_m3, 1) if self.volume_m3 else None,
            "threat": self.threat,
            "url": self.url,
            "is_petroleum": self.is_petroleum,
        }


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[: len(fmt) + 2], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    try:
        result = float(str(value).replace(",", "").strip())
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_noaa_incidents(
    path: str | Path, petroleum_only: bool = True
) -> list[SpillIncident]:
    """Load the NOAA IncidentNews registry.

    Real, documented, responded-to spills - roughly 5,000 of them going back
    to 1957. Coverage is strongest in US waters, which is a limitation to
    state rather than hide.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"NOAA incident registry not found: {p}. "
            f"Run scripts/fetch_incidents.py to download it."
        )

    incidents: list[SpillIncident] = []
    skipped = 0
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            lon, lat = _to_float(row.get("lon")), _to_float(row.get("lat"))
            if lon is None or lat is None:
                skipped += 1
                continue
            # A handful of rows carry unwrapped longitudes (e.g. 237.4).
            if lon > 180.0:
                lon -= 360.0
            elif lon < -180.0:
                lon += 360.0
            if not (-90.0 <= lat <= 90.0):
                skipped += 1
                continue

            incident = SpillIncident(
                incident_id=f"noaa-{row.get('id', '')}",
                name=(row.get("name") or "Unnamed incident").strip(),
                lon=lon, lat=lat,
                occurred_at=_parse_date(row.get("open_date")),
                commodity=(row.get("commodity") or "").strip() or None,
                location=(row.get("location") or "").strip() or None,
                source="NOAA IncidentNews",
                max_release_gallons=_to_float(row.get("max_ptl_release_gallons")),
                threat=(row.get("threat") or "").strip() or None,
                description=(row.get("description") or "").strip()[:500] or None,
                url=(f"https://incidentnews.noaa.gov/incident/{row.get('id')}"
                     if row.get("id") else None),
            )
            if petroleum_only and not incident.is_petroleum:
                skipped += 1
                continue
            incidents.append(incident)

    log.info(
        "Loaded %d petroleum incidents from %s (%d rows skipped)",
        len(incidents), p.name, skipped,
    )
    if not incidents:
        raise ValueError(f"No usable incidents parsed from {p}")
    return incidents


# Major documented spills and persistent leak sites outside the NOAA
# registry's US-centred coverage. Every entry is public record; positions are
# the published incident locations. This exists so the world map is genuinely
# worldwide rather than a map of American waters.
# duration_days: how long the source kept releasing. Present only where the
# figure is a canonical, widely-cited fact. Absent means "treated as a single
# event", which is the conservative reading - it narrows the corroboration
# window rather than widening it, so a missing duration can never manufacture
# a match. Do not guess these.
WORLD_INCIDENTS: list[dict[str, Any]] = [
    # --- Indian waters: our actual target region ---
    dict(id="msc-elsa-3", name="MSC ELSA 3 sinking", lon=76.136, lat=9.3125,
         date="2025-05-25", commodity="furnace oil and diesel",
         location="Off Kochi, Kerala, India",
         # The wreck kept releasing for weeks, so it is a source over that whole
         # span rather than a single event on 25 May. Without this a detection
         # in the middle of the episode is scored as though the oil were three
         # weeks stale.
         duration_days=30,
         note="Container ship sank; wreck leaked for weeks. Our validation case."),
    dict(id="wakashio", name="MV Wakashio grounding", lon=57.665, lat=-20.435,
         date="2020-07-25", commodity="very low sulphur fuel oil",
         location="Pointe d'Esny, Mauritius",
         note="~1,000 t of VLSFO released onto reef; extensively imaged by SAR."),
    dict(id="x-press-pearl", name="X-Press Pearl fire and sinking", lon=79.783, lat=6.983,
         date="2021-05-20", commodity="fuel oil and chemicals",
         location="Off Colombo, Sri Lanka",
         note="Sri Lanka's worst marine disaster; bunker fuel plus nurdles."),
    dict(id="mumbai-2010", name="MSC Chitra / MV Khalijia collision", lon=72.833, lat=18.883,
         date="2010-08-07", commodity="fuel oil",
         location="Off Mumbai, India",
         note="~800 t of fuel oil into Mumbai harbour approaches."),
    dict(id="ennore-2017", name="Ennore oil spill", lon=80.333, lat=13.233,
         date="2017-01-28", commodity="heavy fuel oil",
         location="Ennore, Chennai, India",
         note="Tanker collision off Kamarajar Port."),
    # --- Persistent sources: visible on nearly every satellite pass ---
    dict(id="taylor-mc20", name="Taylor Energy MC-20", lon=-88.970, lat=28.936,
         date="2004-09-16", commodity="crude oil",
         location="Gulf of Mexico, USA", persistent=True,
         note="Leaking continuously since 2004; the longest-running US spill."),
    dict(id="coal-oil-point", name="Coal Oil Point seep field", lon=-119.883, lat=34.391,
         date="1900-01-01", commodity="natural crude seep",
         location="Santa Barbara Channel, USA", persistent=True, natural=True,
         note="Natural seep, not a spill. Present so it is never attributed to a vessel."),
    dict(id="campeche-seeps", name="Campeche Bay natural seeps", lon=-92.200, lat=19.700,
         date="1900-01-01", commodity="natural crude seep",
         location="Southern Gulf of Mexico", persistent=True, natural=True,
         note="Extensive natural seepage. Excluded from vessel attribution."),
    # --- Major world incidents ---
    dict(id="deepwater-horizon", name="Deepwater Horizon", lon=-88.366, lat=28.738,
         date="2010-04-20", commodity="crude oil",
         location="Gulf of Mexico, USA",
         note="~4.9 million barrels; the largest marine oil spill on record.",
         # wellhead flowed 20 Apr - 15 Jul 2010
         duration_days=87),
    dict(id="prestige", name="Prestige tanker sinking", lon=-9.500, lat=42.200,
         date="2002-11-13", commodity="heavy fuel oil",
         location="Off Galicia, Spain"),
    dict(id="erika", name="Erika tanker breakup", lon=-4.250, lat=47.167,
         date="1999-12-12", commodity="heavy fuel oil",
         location="Bay of Biscay, France"),
    dict(id="sanchi", name="Sanchi tanker collision", lon=124.950, lat=28.380,
         date="2018-01-06", commodity="condensate and bunker fuel",
         location="East China Sea"),
    dict(id="exxon-valdez", name="Exxon Valdez", lon=-146.883, lat=60.840,
         date="1989-03-24", commodity="crude oil",
         location="Prince William Sound, Alaska, USA"),
    dict(id="gulf-war", name="Gulf War oil spill", lon=48.500, lat=28.500,
         date="1991-01-19", commodity="crude oil",
         location="Persian Gulf, off Kuwait"),
    dict(id="niger-delta", name="Niger Delta chronic pollution", lon=6.500, lat=4.500,
         date="1970-01-01", commodity="crude oil",
         location="Niger Delta, Nigeria", persistent=True,
         note="Decades of chronic pipeline and wellhead releases."),
    dict(id="bohai-2011", name="Penglai 19-3 platform leak", lon=120.100, lat=38.400,
         date="2011-06-04", commodity="crude oil",
         location="Bohai Bay, China"),
    dict(id="montara", name="Montara wellhead blowout", lon=124.533, lat=-12.683,
         date="2009-08-21", commodity="crude oil and condensate",
         location="Timor Sea, Australia",
         # wellhead flowed 21 Aug - 3 Nov 2009
         duration_days=74),
    dict(id="agia-zoni", name="Agia Zoni II sinking", lon=23.583, lat=37.933,
         date="2017-09-10", commodity="fuel oil",
         location="Saronic Gulf, Greece"),
    dict(id="kerch-2007", name="Volgoneft-139 breakup", lon=36.633, lat=45.283,
         date="2007-11-11", commodity="fuel oil",
         location="Kerch Strait, Black Sea"),
    dict(id="principe-2024", name="Trinidad Gulfstream barge spill", lon=-61.450, lat=10.100,
         date="2024-02-07", commodity="fuel oil",
         location="Tobago, Trinidad and Tobago"),
    # --- Europe / Mediterranean / Black Sea ---
    dict(id="amoco-cadiz", name="Amoco Cadiz grounding", lon=-4.767, lat=48.583,
         date="1978-03-16", commodity="crude oil", location="Brittany, France"),
    dict(id="sea-empress", name="Sea Empress grounding", lon=-5.100, lat=51.683,
         date="1996-02-15", commodity="crude oil", location="Milford Haven, Wales, UK"),
    dict(id="braer", name="Braer grounding", lon=-1.283, lat=59.883,
         date="1993-01-05", commodity="crude oil", location="Shetland, Scotland, UK"),
    dict(id="haven-1991", name="Haven tanker explosion", lon=8.750, lat=44.383,
         date="1991-04-11", commodity="crude oil", location="Off Genoa, Italy"),
    dict(id="lebanon-2006", name="Jiyeh power station spill", lon=35.400, lat=33.650,
         date="2006-07-13", commodity="fuel oil", location="Jiyeh, Lebanon"),
    dict(id="rena-nz", name="MV Rena grounding", lon=176.433, lat=-37.533,
         date="2011-10-05", commodity="heavy fuel oil", location="Astrolabe Reef, New Zealand"),
    dict(id="baltic-butinge", name="Butinge terminal releases", lon=21.050, lat=56.033,
         date="2001-11-01", commodity="crude oil", location="Butinge, Lithuania"),
    dict(id="tricolor", name="MV Tricolor sinking", lon=1.900, lat=51.150,
         date="2002-12-14", commodity="bunker fuel", location="Dover Strait"),
    # --- Asia ---
    dict(id="hebei-spirit", name="Hebei Spirit collision", lon=126.050, lat=36.900,
         date="2007-12-07", commodity="crude oil", location="Taean, South Korea"),
    dict(id="dalian-2010", name="Dalian pipeline explosion", lon=121.900, lat=38.950,
         date="2010-07-16", commodity="crude oil", location="Dalian, China"),
    dict(id="qingdao-2013", name="Huangdao pipeline blast", lon=120.200, lat=36.000,
         date="2013-11-22", commodity="crude oil", location="Qingdao, China"),
    dict(id="singapore-2017", name="Agile / Dominia collision", lon=103.750, lat=1.220,
         date="2017-09-03", commodity="marine fuel oil", location="Singapore Strait"),
    dict(id="singapore-2024", name="Marine Honour allision", lon=103.760, lat=1.253,
         date="2024-06-14", commodity="low-sulphur fuel oil", location="Pasir Panjang, Singapore"),
    dict(id="guimaras", name="Solar 1 sinking", lon=122.583, lat=10.550,
         date="2006-08-11", commodity="bunker fuel", location="Guimaras, Philippines"),
    dict(id="oriental-mindoro", name="MT Princess Empress sinking", lon=121.300, lat=13.283,
         date="2023-02-28", commodity="industrial fuel oil", location="Oriental Mindoro, Philippines"),
    dict(id="sundarbans", name="Southern Star VII sinking", lon=89.600, lat=22.150,
         date="2014-12-09", commodity="furnace oil", location="Sundarbans, Bangladesh"),
    dict(id="karachi-2003", name="Tasman Spirit grounding", lon=66.967, lat=24.800,
         date="2003-07-27", commodity="crude oil", location="Karachi, Pakistan"),
    dict(id="japan-nakhodka", name="Nakhodka breakup", lon=135.000, lat=37.000,
         date="1997-01-02", commodity="heavy fuel oil", location="Sea of Japan"),
    dict(id="wakayama-2021", name="Wakayama refinery leak", lon=135.150, lat=34.183,
         date="2021-04-01", commodity="crude oil", location="Wakayama, Japan"),
    dict(id="persian-gulf-seeps", name="Persian Gulf chronic discharges", lon=51.500, lat=27.500,
         date="1980-01-01", commodity="crude oil", location="Persian Gulf", persistent=True,
         note="Chronic operational discharges along one of the busiest tanker routes."),
    dict(id="malacca-strait", name="Malacca Strait chronic discharges", lon=100.500, lat=3.500,
         date="1990-01-01", commodity="oily waste", location="Strait of Malacca", persistent=True,
         note="Routine bilge and tank-washing discharges on a major shipping lane."),
    # --- Africa / Middle East ---
    dict(id="bonga-2011", name="Bonga FPSO spill", lon=5.100, lat=4.500,
         date="2011-12-20", commodity="crude oil", location="Offshore Nigeria"),
    dict(id="bodo-2008", name="Bodo pipeline spills", lon=7.283, lat=4.617,
         date="2008-08-28", commodity="crude oil", location="Ogoniland, Nigeria"),
    dict(id="jebel-ali-2010", name="Gulf of Oman sheen", lon=56.500, lat=25.500,
         date="2010-08-01", commodity="crude oil", location="Gulf of Oman"),
    dict(id="safer-tanker", name="FSO Safer risk site", lon=42.700, lat=15.117,
         date="2015-01-01", commodity="crude oil", location="Off Ras Isa, Yemen",
         note="Decaying floating storage vessel; cargo transferred 2023."),
    dict(id="suez-2010", name="Jebel al-Zayt platform leak", lon=33.583, lat=27.783,
         date="2010-06-16", commodity="crude oil", location="Red Sea, Egypt"),
    dict(id="cape-town-2019", name="Table Bay bunkering spill", lon=18.417, lat=-33.900,
         date="2019-07-05", commodity="marine fuel oil", location="Cape Town, South Africa"),
    dict(id="algoa-bay", name="Algoa Bay bunkering releases", lon=25.700, lat=-33.950,
         date="2019-01-01", commodity="marine fuel oil", location="Algoa Bay, South Africa",
         persistent=True, note="Repeated small releases from ship-to-ship bunkering."),
    # --- Americas beyond the NOAA registry ---
    dict(id="peru-repsol", name="La Pampilla refinery spill", lon=-77.150, lat=-11.900,
         date="2022-01-15", commodity="crude oil", location="Ventanilla, Peru",
         note="Tanker discharge during a swell surge; major Pacific coast spill."),
    dict(id="brazil-2019", name="Northeast Brazil mystery spill", lon=-37.000, lat=-10.000,
         date="2019-08-30", commodity="crude oil", location="Northeast Brazil coast",
         note="Thousands of km of coastline oiled; source long disputed."),
    dict(id="campos-2011", name="Frade field leak", lon=-40.100, lat=-22.400,
         date="2011-11-07", commodity="crude oil", location="Campos Basin, Brazil"),
    dict(id="venezuela-2020", name="El Palito refinery leak", lon=-68.150, lat=10.483,
         date="2020-07-22", commodity="crude oil", location="Carabobo, Venezuela"),
    dict(id="argentina-1999", name="Estrella Pampeana collision", lon=-57.900, lat=-34.783,
         date="1999-01-15", commodity="crude oil", location="Rio de la Plata, Argentina"),
    dict(id="newfoundland-2018", name="SeaRose FPSO release", lon=-48.500, lat=46.750,
         date="2018-11-16", commodity="crude oil", location="Grand Banks, Canada"),
]


def load_world_incidents() -> list[SpillIncident]:
    """The curated worldwide catalogue. Public record, every entry sourced."""
    out: list[SpillIncident] = []
    for entry in WORLD_INCIDENTS:
        out.append(SpillIncident(
            incident_id=f"world-{entry['id']}",
            name=entry["name"],
            lon=float(entry["lon"]), lat=float(entry["lat"]),
            occurred_at=_parse_date(entry["date"]),
            commodity=entry.get("commodity"),
            location=entry.get("location"),
            source="curated world catalogue",
            description=entry.get("note"),
            extras={
                "persistent": bool(entry.get("persistent", False)),
                "natural_seep": bool(entry.get("natural", False)),
                # How long the source kept releasing. Zero means a single
                # event; a wreck that leaks for weeks is a source for weeks,
                # and the corroboration window has to reflect that or the
                # middle of the episode scores as stale oil.
                "duration_days": float(entry.get("duration_days", 0.0) or 0.0),
            },
        ))
    return out


@dataclass
class Corroboration:
    """An independent registry entry supporting one detection."""

    incident: SpillIncident
    distance_km: float
    days_apart: float | None
    confidence: float          # 0-1
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident": self.incident.to_dict(),
            "distance_km": round(self.distance_km, 1),
            "days_apart": round(self.days_apart, 2) if self.days_apart is not None else None,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


class IncidentRegistry:
    """Spatial and temporal lookup over the documented incidents."""

    def __init__(self, incidents: Iterable[SpillIncident]) -> None:
        self.incidents = list(incidents)
        # Coarse 5-degree spatial index. A linear scan over several thousand
        # incidents per detection is wasteful when a map pan triggers many.
        self._grid: dict[tuple[int, int], list[SpillIncident]] = {}
        for incident in self.incidents:
            self._grid.setdefault(self._cell(incident.lon, incident.lat), []).append(incident)

    @staticmethod
    def _cell(lon: float, lat: float) -> tuple[int, int]:
        return (int(math.floor(lon / 5.0)), int(math.floor(lat / 5.0)))

    def _nearby(self, lon: float, lat: float) -> list[SpillIncident]:
        cx, cy = self._cell(lon, lat)
        out: list[SpillIncident] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out.extend(self._grid.get((cx + dx, cy + dy), []))
        return out

    def find(
        self,
        lon: float,
        lat: float,
        when: datetime | None = None,
        base_radius_km: float = CORROBORATION_BASE_KM,
    ) -> list[Corroboration]:
        """Documented incidents that could explain a detection here and then.

        Returns them ranked by confidence, strongest first.
        """
        matches: list[Corroboration] = []
        for incident in self._nearby(lon, lat):
            distance = haversine_km((lon, lat), (incident.lon, incident.lat))
            persistent = bool(incident.extras.get("persistent"))
            days_apart: float | None = None

            # An incident that keeps releasing is a source for as long as it
            # does so. A wreck leaking for weeks is not "one event on day zero"
            # followed by silence, and scoring it that way rejects the middle of
            # the very episode being validated.
            duration_days = float(incident.extras.get("duration_days", 0.0) or 0.0)

            if when is not None and incident.occurred_at is not None and not persistent:
                delta_days = (when - incident.occurred_at).total_seconds() / 86400.0
                days_apart = delta_days
                # A detection BEFORE the incident cannot be that incident;
                # long after the source stops, the slick has dispersed.
                if delta_days < -CORROBORATION_DAYS_BEFORE:
                    continue
                if delta_days > CORROBORATION_DAYS_AFTER + duration_days:
                    continue

                # Age is measured from when the source stopped releasing, not
                # from when it started: oil put in the water on day 14 of a leak
                # is fresh on day 14.
                age_days = max(delta_days - duration_days, 0.0)
                # Drift is allowed for from the START, because the earliest oil
                # has been moving the whole time.
                radius_km = corroboration_radius_km(delta_days, base_radius_km)
                time_term = math.exp(-age_days / 7.0)
                reason = (
                    f"{distance:.0f} km from the documented '{incident.name}' "
                    f"({incident.occurred_at:%d %b %Y}), {abs(delta_days):.1f} days "
                    f"{'after' if delta_days >= 0 else 'before'}"
                    + (f"; source released for ~{duration_days:.0f} days"
                       if duration_days else "")
                    + (f"; within {radius_km:.0f} km of plausible drift"
                       if delta_days > 0 else "")
                )
            elif persistent:
                radius_km = max(base_radius_km, CORROBORATION_PERSISTENT_KM)
                time_term = 1.0
                reason = (
                    f"{distance:.0f} km from '{incident.name}', a known persistent "
                    f"source that appears on nearly every pass"
                )
            else:
                radius_km = base_radius_km
                time_term = 0.4
                reason = (
                    f"{distance:.0f} km from the documented '{incident.name}' "
                    f"(no usable date to compare)"
                )

            if distance > radius_km:
                continue

            distance_term = max(0.0, 1.0 - distance / max(radius_km, 1e-6))
            matches.append(Corroboration(
                incident=incident,
                distance_km=distance,
                days_apart=days_apart,
                confidence=round(0.6 * distance_term + 0.4 * time_term, 4),
                reason=reason,
            ))

        matches.sort(key=lambda c: c.confidence, reverse=True)
        return matches

    def __len__(self) -> int:
        return len(self.incidents)


def load_registry(
    noaa_path: str | Path | None = None, include_world: bool = True
) -> IncidentRegistry:
    """Build the combined registry from every source available."""
    incidents: list[SpillIncident] = []
    if noaa_path:
        try:
            incidents.extend(load_noaa_incidents(noaa_path))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("NOAA registry unavailable (%s); using the world catalogue only", exc)
    if include_world:
        incidents.extend(load_world_incidents())
    if not incidents:
        raise ValueError(
            "No incident sources loaded. Run scripts/fetch_incidents.py."
        )
    log.info("Incident registry: %d documented spills", len(incidents))
    return IncidentRegistry(incidents)
