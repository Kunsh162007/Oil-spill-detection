"""Geodesy helpers. Everything is (lon, lat) — see core.contracts.

Kept deliberately small and dependency-light: scoring, drift and morphology
all need these, and a pyproj round-trip per particle per timestep is far too
slow for the drift loop.
"""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0088


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lon, lat) points."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Initial great-circle bearing a -> b, degrees clockwise from north."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, 0-180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def axial_difference_deg(a: float, b: float) -> float:
    """Smallest difference treating the directions as *undirected* axes, 0-90.

    A slick's long axis has no head or tail, so a ship steaming 090 and one
    steaming 270 are equally parallel to it. Using bearing_deg here instead
    would score the reciprocal course as maximally *un*aligned.
    """
    return abs((a - b + 90.0) % 180.0 - 90.0)


def destination_point(
    origin: tuple[float, float], bearing: float, distance_km: float
) -> tuple[float, float]:
    """Point reached travelling distance_km from origin along bearing."""
    lon1, lat1 = math.radians(origin[0]), math.radians(origin[1])
    brg = math.radians(bearing)
    ang = distance_km / EARTH_RADIUS_KM
    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang) + math.cos(lat1) * math.sin(ang) * math.cos(brg)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lon2 + math.pi) % 360.0 - 180.0, math.degrees(lat2))


def offset_by_metres(
    origin: tuple[float, float], east_m: float, north_m: float
) -> tuple[float, float]:
    """Shift a (lon, lat) point by an east/north offset in metres.

    Local flat-earth approximation. Valid for the few-km displacements a
    drift timestep produces; do not use it for basin-scale hops.
    """
    lon, lat = origin
    dlat = north_m / 110_574.0
    denom = 111_320.0 * math.cos(math.radians(lat))
    dlon = east_m / denom if abs(denom) > 1e-9 else 0.0
    return (lon + dlon, lat + dlat)


def bbox_contains(bbox: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon, lat = point
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def expand_bbox(
    bbox: tuple[float, float, float, float], km: float
) -> tuple[float, float, float, float]:
    """Grow a bbox by roughly `km` in every direction, clamped to valid ranges."""
    min_lon, min_lat, max_lon, max_lat = bbox
    dlat = km / 110.574
    mid_lat = (min_lat + max_lat) / 2.0
    denom = 111.320 * math.cos(math.radians(mid_lat))
    dlon = km / denom if abs(denom) > 1e-9 else 180.0
    return (
        max(-180.0, min_lon - dlon),
        max(-90.0, min_lat - dlat),
        min(180.0, max_lon + dlon),
        min(90.0, max_lat + dlat),
    )
