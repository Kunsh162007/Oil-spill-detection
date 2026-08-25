"""Confidence tiering - how sure are we this is actually oil?

P(oil) alone is one model's opinion. This combines it with evidence that
comes from OUTSIDE the segmentation network, so the answer to "is this a real
spill" does not rest on a single detector:

  * how far inside the SAR wind window the observation sits
  * how strongly the patch damps backscatter, in dB
  * whether the shape matches a discharge or a look-alike
  * whether an independent incident registry recorded a spill here
  * whether the slick sits on a catalogued seep or installation

Four tiers, and the UI shows only the top two by default. CLAUDE.md's whole
premise is that a confident wrong call costs more than an honest shrug, so a
detection that cannot clear the bar is kept and labelled rather than shown as
oil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["confirmed", "probable", "possible", "insufficient"]

TIER_ORDER: dict[Tier, int] = {
    "confirmed": 3, "probable": 2, "possible": 1, "insufficient": 0,
}

TIER_MEANING: dict[Tier, str] = {
    "confirmed": (
        "Physics indicates oil AND an independent incident registry records a "
        "spill at this place and time."
    ),
    "probable": (
        "Strong physical evidence of oil: good wind conditions, clear "
        "backscatter damping, and a discharge-like shape."
    ),
    "possible": (
        "Consistent with oil but not strongly supported. Shown only when "
        "low-confidence detections are explicitly enabled."
    ),
    "insufficient": (
        "Does not meet the bar for a real slick. Retained with its reason so "
        "the rejection can be inspected."
    ),
}


@dataclass
class ConfidenceAssessment:
    tier: Tier
    score: float                      # 0-1, continuous behind the tier
    reasons: list[str] = field(default_factory=list)
    corroborated: bool = False

    @property
    def is_actual_oil(self) -> bool:
        """Whether to present this as a real spill by default."""
        return TIER_ORDER[self.tier] >= TIER_ORDER["probable"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "corroborated": self.corroborated,
            "meaning": TIER_MEANING[self.tier],
            "is_actual_oil": self.is_actual_oil,
        }


# Thresholds. Deliberately conservative: the cost of showing a look-alike as
# a confirmed spill is a wrongly accused ship.
STRONG_P_OIL = 0.75
MIN_P_OIL = 0.50
STRONG_DAMPING_DB = 5.0
# Independent confirmation is worth a lot - it comes from outside the model
# entirely - but it is capped so it can never carry a candidate on its own, and
# it is scaled by how good the match is. The corroboration radius grows with
# drift time, so a match 200 km out after a fortnight is real evidence but
# weaker than one sitting on the incident.
CORROBORATION_BONUS = 0.30
# Below this, a match still helps the score but may not promote to "confirmed".
MIN_CORROBORATION_CONFIDENCE = 0.35

MIN_DAMPING_DB = 2.5
STRONG_WIND_WINDOW = 0.6
MIN_WIND_WINDOW = 0.25
STRONG_ELONGATION = 8.0
MIN_AREA_KM2 = 0.3


def _damping_db(damping_ratio: float | None) -> float:
    if damping_ratio is None or not math.isfinite(damping_ratio) or damping_ratio <= 0:
        return 0.0
    return -10.0 * math.log10(damping_ratio)


def assess(
    p_oil: float,
    wind_window_score: float,
    damping_ratio: float | None,
    elongation: float,
    area_km2: float,
    morphology: str,
    rejected_reason: str | None = None,
    corroboration: dict[str, Any] | None = None,
    source_type: str = "unknown",
) -> ConfidenceAssessment:
    """Grade one candidate into a confidence tier, with its reasoning."""
    reasons: list[str] = []
    corroboration = corroboration or {}
    corroborated = bool(corroboration.get("confirmed"))

    # A physics rejection settles it. Corroboration cannot resurrect a patch
    # the physics says could not be oil - the two would be measuring
    # different things, and the registry has no view of this pixel.
    if rejected_reason:
        return ConfidenceAssessment(
            tier="insufficient", score=0.0,
            reasons=[rejected_reason], corroborated=corroborated,
        )

    score = 0.0

    # Physics terms.
    if p_oil >= STRONG_P_OIL:
        score += 0.34
        reasons.append(f"P(oil) {p_oil:.2f} is strong")
    elif p_oil >= MIN_P_OIL:
        score += 0.18
        reasons.append(f"P(oil) {p_oil:.2f} is moderate")
    else:
        reasons.append(f"P(oil) {p_oil:.2f} is weak")

    damping = _damping_db(damping_ratio)
    if damping >= STRONG_DAMPING_DB:
        score += 0.22
        reasons.append(f"damps backscatter by {damping:.1f} dB, typical of mineral oil")
    elif damping >= MIN_DAMPING_DB:
        score += 0.11
        reasons.append(f"damping {damping:.1f} dB is modest")
    else:
        reasons.append(f"damping {damping:.1f} dB is weak for a real slick")

    if wind_window_score >= STRONG_WIND_WINDOW:
        score += 0.20
        reasons.append("wind sits well inside the SAR detection window")
    elif wind_window_score >= MIN_WIND_WINDOW:
        score += 0.09
        reasons.append("wind is near the edge of the detection window")
    else:
        reasons.append("wind is at the limit of where SAR can see oil at all")

    if morphology == "linear" and elongation >= STRONG_ELONGATION:
        score += 0.14
        reasons.append(f"elongation {elongation:.0f}:1 matches a moving-vessel discharge")
    elif morphology == "linear":
        score += 0.07
        reasons.append("shape is broadly linear")

    if area_km2 >= MIN_AREA_KM2:
        score += 0.10
    else:
        reasons.append(f"area {area_km2:.2f} km2 is small enough to be residual speckle")

    # Independent confirmation. Worth a lot, because it comes from outside the
    # model entirely - but it is capped so it can never carry a candidate on
    # its own.
    match = (corroboration.get("matches") or [{}])[0]
    match_confidence = float(match.get("confidence", 0.0) or 0.0)

    if corroborated:
        # Scaled by how good the match is, not flat. The corroboration radius
        # now grows with drift time, so a match can legitimately be 200 km away
        # after a fortnight - real evidence, but weaker than one sitting on top
        # of the incident, and it must not score the same.
        weight = min(1.0, 0.4 + match_confidence)
        score = min(1.0, score + CORROBORATION_BONUS * weight)
        reasons.append(
            "independently corroborated: " + str(match.get("reason", "documented incident nearby"))
        )

    # Only a strong match may promote to "confirmed"; a distant one still helps
    # the score but cannot, by itself, upgrade the verdict.
    strong_corroboration = corroborated and match_confidence >= MIN_CORROBORATION_CONFIDENCE

    tier: Tier
    if strong_corroboration and score >= 0.55:
        tier = "confirmed"
    elif score >= 0.62:
        tier = "probable"
    elif score >= 0.38:
        tier = "possible"
    else:
        tier = "insufficient"

    # A fixed source is real oil, but it is never a vessel discharge. Grading
    # it as merely "possible" would hide a genuine, persistent spill.
    if source_type in ("natural_seep", "infrastructure") and tier == "possible":
        tier = "probable"
        reasons.append(f"matches a catalogued {source_type.replace('_', ' ')}")

    return ConfidenceAssessment(
        tier=tier, score=min(score, 1.0), reasons=reasons, corroborated=corroborated
    )
