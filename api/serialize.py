"""Turn pipeline objects into the GeoJSON the map consumes.

The UI never sees a dataclass. Everything crossing this boundary is plain
JSON with lon/lat in GeoJSON order, and every caveat the pipeline recorded
travels with it - the disclaimers are part of the payload, not decoration
the front end can forget to add.
"""

from __future__ import annotations

import math
from typing import Any

from core.contracts import Attribution, SlickCandidate
from decision.pipeline import SceneAnalysis
from decision.rank import DISCLAIMER


def _wkt_to_coords(wkt: str) -> list[list[float]]:
    """WKT POLYGON -> a GeoJSON linear ring."""
    if not wkt or wkt.upper().startswith("POLYGON EMPTY"):
        return []
    try:
        from shapely import wkt as shapely_wkt

        geom = shapely_wkt.loads(wkt)
        if geom.is_empty:
            return []
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        return [[round(float(x), 6), round(float(y), 6)] for x, y in geom.exterior.coords]
    except Exception:
        return []


def _clean(value: Any) -> Any:
    """NaN and Infinity are not valid JSON; browsers choke on them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


# A detection is only "active" while a present position can honestly be
# forecast. Past that it is a past incident: real, but historical, and it
# should not sit on the map implying oil is there right now.
ACTIVE_WINDOW_HOURS = 72.0


def scene_status(acquired_at, now=None) -> tuple[str, float]:
    """('active'|'historical', age_hours) for a scene acquisition time."""
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    age = (now - acquired_at).total_seconds() / 3600.0
    return ("active" if age <= ACTIVE_WINDOW_HOURS else "historical"), age


def slick_feature(
    candidate: SlickCandidate,
    attribution: Attribution | None,
    scene_id: str,
    acquired_at=None,
) -> dict[str, Any]:
    """One slick as a GeoJSON Feature carrying its full evidence."""
    ring = _wkt_to_coords(candidate.polygon_wkt)
    geometry = (
        {"type": "Polygon", "coordinates": [ring]}
        if len(ring) >= 4
        else {"type": "Point", "coordinates": list(candidate.centroid)}
    )

    status = (
        "rejected" if candidate.is_rejected
        else "abstained" if (attribution and attribution.abstained)
        else "attributed"
    )

    props: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "scene_id": scene_id,
        "status": status,
        "is_oil": not candidate.is_rejected,
        "p_oil": _clean(candidate.p_oil),
        "rejected_reason": candidate.rejected_reason,
        "area_km2": _clean(candidate.area_km2),
        "elongation": _clean(candidate.elongation),
        "compactness": _clean(candidate.compactness),
        "damping_ratio": _clean(candidate.damping_ratio),
        "morphology": candidate.morphology,
        "centroid": [round(candidate.centroid[0], 6), round(candidate.centroid[1], 6)],
        "wind": {
            "speed_ms": _clean(round(candidate.wind.speed_ms, 2)),
            "direction_deg": _clean(round(candidate.wind.direction_deg, 1)),
            "source": candidate.wind.source,
            "window_score": _clean(round(candidate.wind.window_score, 3)),
        },
        "texture": {
            "homogeneity": _clean(candidate.texture_homogeneity),
            "contrast": _clean(candidate.texture_contrast),
        },
        "vh_vv_ratio": _clean(candidate.vh_vv_ratio),
        "physics_contributions": {
            k: _clean(v) for k, v in candidate.feature_contributions.items()
        },
        "disclaimer": DISCLAIMER,
    }

    if acquired_at is not None:
        status, age_hours = scene_status(acquired_at)
        props["acquired_at"] = acquired_at.isoformat()
        props["age_hours"] = round(age_hours, 1)
        props["age_days"] = round(age_hours / 24.0, 1)
        props["activity"] = status

    if attribution is not None:
        # Independent confirmation from an incident registry - the strongest
        # signal that a detection is a real spill rather than a look-alike.
        corroboration = attribution.evidence.get("corroboration") or {}
        props["corroborated"] = bool(corroboration.get("confirmed"))
        props["corroboration"] = (
            corroboration.get("matches", [])[:1] if corroboration.get("confirmed") else []
        )
        confidence = attribution.evidence.get("confidence") or {}
        props["confidence_tier"] = confidence.get("tier", "unknown")
        props["confidence_score"] = confidence.get("score")
        props["confidence_reasons"] = confidence.get("reasons", [])
        props["is_actual_oil"] = bool(confidence.get("is_actual_oil"))
        props["source_type"] = attribution.source_type
        props["abstained"] = attribution.abstained
        props["abstain_reason"] = attribution.abstain_reason
        props["n_candidates"] = len(attribution.candidates)
        props["top_candidate"] = (
            {
                "mmsi": attribution.candidates[0].mmsi,
                "name": attribution.candidates[0].name,
                "score": _clean(attribution.candidates[0].score),
                "went_dark": attribution.candidates[0].went_dark,
            }
            if attribution.candidates and not attribution.abstained
            else None
        )
        if attribution.origin is not None:
            props["origin"] = {
                "lon": round(attribution.origin.lon, 6),
                "lat": round(attribution.origin.lat, 6),
                "estimated_at": attribution.origin.estimated_at.isoformat(),
                "uncertainty_km": _clean(attribution.origin.uncertainty_km),
                "method": attribution.origin.method,
            }

    return {"type": "Feature", "geometry": geometry, "properties": props}


def scene_collection(analysis: SceneAnalysis) -> dict[str, Any]:
    """Every slick in a scene as a GeoJSON FeatureCollection."""
    by_id = {a.candidate_id: a for a in analysis.attributions}
    features = [
        slick_feature(c, by_id.get(c.candidate_id), analysis.scene.scene_id,
                      analysis.scene.acquired_at)
        for c in analysis.candidates
    ]
    return {
        "type": "FeatureCollection",
        "features": features,
        "scene": {
            "scene_id": analysis.scene.scene_id,
            "acquired_at": analysis.scene.acquired_at.isoformat(),
            "bbox": list(analysis.scene.bbox),
            "orbit_direction": analysis.scene.orbit_direction,
            "dual_pol": analysis.scene.is_dual_pol,
        },
        "stats": analysis.stats,
        "timings": analysis.timings,
        "warnings": analysis.warnings,
    }


def attribution_detail(
    candidate: SlickCandidate, attribution: Attribution
) -> dict[str, Any]:
    """The full detail view: origin, drift track, ranked vessels, evidence."""
    origin = attribution.origin
    detail: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "source_type": attribution.source_type,
        "abstained": attribution.abstained,
        "abstain_reason": attribution.abstain_reason,
        "disclaimer": DISCLAIMER,
        "slick": {
            "polygon": _wkt_to_coords(candidate.polygon_wkt),
            "centroid": list(candidate.centroid),
            "area_km2": _clean(candidate.area_km2),
            "p_oil": _clean(candidate.p_oil),
            "morphology": candidate.morphology,
            "rejected_reason": candidate.rejected_reason,
        },
        "wind": {
            "speed_ms": _clean(round(candidate.wind.speed_ms, 2)),
            "direction_deg": _clean(round(candidate.wind.direction_deg, 1)),
            "source": candidate.wind.source,
            "window_score": _clean(round(candidate.wind.window_score, 3)),
        },
        "evidence": attribution.evidence,
        "vessels": [],
    }

    if origin is not None:
        detail["origin"] = {
            "lon": round(origin.lon, 6),
            "lat": round(origin.lat, 6),
            "estimated_at": origin.estimated_at.isoformat(),
            "uncertainty_km": _clean(origin.uncertainty_km),
            "backtrack_hours": _clean(origin.backtrack_hours),
            "n_particles": origin.n_particles,
            "method": origin.method,
            "reliable": origin.is_reliable,
            # Oldest first - the UI plays this forward so the animation runs
            # the way the oil actually drifted.
            "track": origin.track,
        }

    for rank, v in enumerate(attribution.candidates, start=1):
        voyage = (attribution.evidence.get("voyages") or {}).get(v.mmsi)
        detail["vessels"].append({
            "voyage": voyage,
            "rank": rank,
            "mmsi": v.mmsi,
            "name": v.name,
            "vessel_type": v.vessel_type,
            "flag": v.flag,
            "score": _clean(v.score),
            "parity": _clean(v.parity),
            "proximity": _clean(v.proximity),
            "temporality": _clean(v.temporality),
            "went_dark": v.went_dark,
            "evidence": v.evidence,
            "closest_approach_km": _clean(v.closest_approach_km),
            "closest_approach_at": (
                v.closest_approach_at.isoformat() if v.closest_approach_at else None
            ),
            "track": v.track,
        })

    return detail


def world_index(analyses: list[SceneAnalysis]) -> dict[str, Any]:
    """All confirmed slicks across every analysed scene, for the world map."""
    features: list[dict[str, Any]] = []
    for analysis in analyses:
        by_id = {a.candidate_id: a for a in analysis.attributions}
        for c in analysis.candidates:
            if c.is_rejected:
                continue
            feature = slick_feature(c, by_id.get(c.candidate_id), analysis.scene.scene_id,
                                    analysis.scene.acquired_at)
            tier = feature["properties"].get("confidence_tier")
            # Only "confirmed" and "probable" are presented as actual oil.
            # Lower tiers stay available through the per-scene endpoint.
            if tier in ("confirmed", "probable") or tier == "unknown":
                features.append(feature)
    active = [f for f in features if f["properties"].get("activity") == "active"]
    historical = [f for f in features if f["properties"].get("activity") == "historical"]

    # What the DATA actually used, counted from the analyses themselves. The
    # service's default config is not the answer: it describes the fallback for
    # a scene that names no config of its own.
    wind_sources: dict[str, int] = {}
    current_sources: dict[str, int] = {}
    for analysis in analyses:
        for attribution in analysis.attributions:
            fields = attribution.evidence.get("field_sources") or {}
            if fields.get("wind"):
                wind_sources[fields["wind"]] = wind_sources.get(fields["wind"], 0) + 1
            if fields.get("currents"):
                current_sources[fields["currents"]] =                     current_sources.get(fields["currents"], 0) + 1
        for candidate in analysis.candidates:
            source = getattr(candidate.wind, "source", None)
            if source and not wind_sources:
                wind_sources[source] = wind_sources.get(source, 0) + 1

    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "wind_sources": wind_sources,
            "currents_sources": current_sources,
            "n_scenes": len(analyses),
            "n_slicks": len(features),
            "n_active": len(active),
            "n_historical": len(historical),
            "active_window_hours": ACTIVE_WINDOW_HOURS,
            "disclaimer": DISCLAIMER,
            "timeliness": (
                "Near-real-time: imagery arrives 3-24 h after acquisition and "
                "free AIS lags about 72 h. This is not a live feed."
            ),
        },
    }


def refresh_ages(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute the time-dependent fields on a cached world index.

    The index is built once, at image build time. "Active" means "acquired
    within ACTIVE_WINDOW_HOURS", which is only true at the instant the cache was
    written - serving the frozen flag would tell a viewer that a week-old
    detection is current oil. Acquisition time is immutable and already in every
    feature, so the age fields are recomputed on the way out.

    Rewrites the flags in place, which is safe because the caller parses the
    cache file fresh on every request rather than sharing one dict. That costs
    a small JSON parse each time and buys freedom from a data race: sync
    endpoints run in a threadpool, so a shared payload mutated per request
    would be rewritten by several threads at once.
    """
    from datetime import datetime

    n_active = 0
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        raw = props.get("acquired_at")
        if not raw:
            continue
        status, age_hours = scene_status(datetime.fromisoformat(raw))
        props["age_hours"] = round(age_hours, 1)
        props["age_days"] = round(age_hours / 24.0, 1)
        props["activity"] = status
        if status == "active":
            n_active += 1

    meta = payload.setdefault("meta", {})
    meta["n_active"] = n_active
    meta["n_historical"] = len(payload.get("features", [])) - n_active
    meta["cached"] = True
    return payload
