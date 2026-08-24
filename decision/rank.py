"""Final decision: bundle the evidence, rank, or abstain.

Abstention is a feature, not a failure. The system returns "insufficient
evidence" when:

  * wind was outside the SAR detection window
  * no AIS coverage existed at the origin
  * the top two candidates are within noise of each other
  * drift uncertainty is too large for the origin to mean anything

A confident wrong accusation costs far more than an honest shrug, and a
system that never abstains is not measuring anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.contracts import (
    Attribution,
    DriftOrigin,
    SlickCandidate,
    SourceType,
    VesselCandidate,
)

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Ranked candidates are correlations between a drift-estimated origin and "
    "AIS tracks. They are investigative leads, not evidence of responsibility."
)


@dataclass
class DecisionConfig:
    abstain_margin: float = 0.08     # top-two closer than this => abstain
    min_top_score: float = 0.35
    min_wind_window_score: float = 0.15
    max_uncertainty_km: float = 60.0


def decide(
    candidate: SlickCandidate,
    origin: DriftOrigin | None,
    source_type: SourceType,
    vessels: list[VesselCandidate],
    cfg: DecisionConfig,
    extra_evidence: dict[str, Any] | None = None,
) -> Attribution:
    """Turn scored vessels into a final Attribution, abstaining when unclear."""
    evidence: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "wind": {
            "speed_ms": candidate.wind.speed_ms,
            "direction_deg": candidate.wind.direction_deg,
            "source": candidate.wind.source,
            "window_score": round(candidate.wind.window_score, 3),
        },
        "slick": {
            "candidate_id": candidate.candidate_id,
            "area_km2": round(candidate.area_km2, 3),
            "p_oil": candidate.p_oil,
            "morphology": candidate.morphology,
            "elongation": round(candidate.elongation, 2),
            "damping_ratio": (
                round(candidate.damping_ratio, 4)
                if candidate.damping_ratio == candidate.damping_ratio  # NaN check
                else None
            ),
        },
        "physics_contributions": candidate.feature_contributions,
    }
    if origin is not None:
        evidence["drift"] = {
            "origin_lon": round(origin.lon, 5),
            "origin_lat": round(origin.lat, 5),
            "estimated_at": origin.estimated_at.isoformat(),
            "uncertainty_km": origin.uncertainty_km,
            "backtrack_hours": origin.backtrack_hours,
            "method": origin.method,
            "n_particles": origin.n_particles,
            "reliable": origin.is_reliable,
        }
    if extra_evidence:
        evidence.update(extra_evidence)

    reason = _abstain_reason(candidate, origin, source_type, vessels, cfg)
    if reason is not None:
        log.info("Abstaining on %s: %s", candidate.candidate_id, reason)
        return Attribution(
            candidate_id=candidate.candidate_id,
            origin=origin,
            source_type=source_type,
            candidates=vessels,   # still returned, clearly marked abstained
            abstained=True,
            abstain_reason=reason,
            evidence=evidence,
        )

    if len(vessels) >= 2:
        evidence["margin"] = round(vessels[0].score - vessels[1].score, 4)

    return Attribution(
        candidate_id=candidate.candidate_id,
        origin=origin,
        source_type=source_type,
        candidates=vessels,
        abstained=False,
        abstain_reason=None,
        evidence=evidence,
    )


def _abstain_reason(
    candidate: SlickCandidate,
    origin: DriftOrigin | None,
    source_type: SourceType,
    vessels: list[VesselCandidate],
    cfg: DecisionConfig,
) -> str | None:
    """The single clearest reason to withhold a ranking, or None."""

    # A rejected look-alike never reaches vessel attribution at all.
    if candidate.rejected_reason:
        return f"not classified as oil: {candidate.rejected_reason}"

    if candidate.wind.window_score < cfg.min_wind_window_score:
        return (
            f"wind {candidate.wind.speed_ms:.1f} m/s sits at the edge of the SAR "
            f"detection window (suitability {candidate.wind.window_score:.2f}) - "
            f"detection itself is unreliable here, so attribution is withheld"
        )

    # Fixed sources are excluded from vessel attribution outright. Blaming a
    # ship for a natural seep is the worst failure this system can produce.
    if source_type in ("natural_seep", "infrastructure"):
        return (
            f"source classified as {source_type.replace('_', ' ')} - vessel "
            f"attribution deliberately not attempted"
        )

    if origin is None:
        return "no drift origin could be computed - cannot ask who was there"

    if origin.uncertainty_km > cfg.max_uncertainty_km:
        return (
            f"drift origin uncertain to +/-{origin.uncertainty_km:.0f} km, beyond the "
            f"{cfg.max_uncertainty_km:.0f} km limit - the origin does not constrain "
            f"which vessel was present"
        )

    if not vessels:
        return (
            "no AIS tracks near the estimated origin in the query window - "
            "either no vessel was present, or AIS coverage is absent there "
            "(free AIS lags ~72 h)"
        )

    top = vessels[0]
    if top.score < cfg.min_top_score:
        return (
            f"best candidate scores only {top.score:.2f}, below the "
            f"{cfg.min_top_score:.2f} threshold - no vessel fits the origin well"
        )

    if len(vessels) >= 2:
        margin = top.score - vessels[1].score
        if margin < cfg.abstain_margin:
            return (
                f"top two candidates are within {margin:.3f} "
                f"({top.name or top.mmsi} {top.score:.2f} vs "
                f"{vessels[1].name or vessels[1].mmsi} {vessels[1].score:.2f}) - "
                f"too close to separate, insufficient evidence"
            )

    return None


def coverage_risk(attributions: list[Attribution]) -> dict[str, Any]:
    """Abstention rate and the resulting coverage, for the evaluation report."""
    total = len(attributions)
    if total == 0:
        return {"total": 0, "abstained": 0, "coverage": 0.0, "reasons": {}}

    abstained = [a for a in attributions if a.abstained]
    reasons: dict[str, int] = {}
    for a in abstained:
        key = (a.abstain_reason or "unknown").split(" - ")[0].split(":")[0][:60]
        reasons[key] = reasons.get(key, 0) + 1

    return {
        "total": total,
        "abstained": len(abstained),
        "coverage": round(1.0 - len(abstained) / total, 4),
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
