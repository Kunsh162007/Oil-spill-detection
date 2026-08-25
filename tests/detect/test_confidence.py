"""Confidence tiering - the guard that only actual oil is presented as oil."""

from __future__ import annotations

import pytest

from detect.confidence import TIER_ORDER, assess


def real_slick(**overrides):
    base = dict(
        p_oil=0.93, wind_window_score=1.0, damping_ratio=0.16,
        elongation=35.0, area_km2=52.0, morphology="linear",
    )
    base.update(overrides)
    return base


CORROBORATION = {
    "confirmed": True,
    # A real match always carries a confidence; the tier logic reads it, so a
    # fixture without one tests a shape that never occurs.
    "matches": [{"reason": "3 km from the documented 'MSC ELSA 3 sinking'",
                 "confidence": 0.92}],
}

# Same incident, but matched at the far edge of the drift-scaled radius - the
# kind of match that only became possible once the radius grew with time.
DISTANT_CORROBORATION = {
    "confirmed": True,
    "matches": [{"reason": "210 km from the documented 'MSC ELSA 3 sinking'",
                 "confidence": 0.11}],
}


class TestTiers:
    def test_strong_physics_plus_registry_is_confirmed(self):
        a = assess(**real_slick(), corroboration=CORROBORATION)
        assert a.tier == "confirmed"
        assert a.corroborated and a.is_actual_oil

    def test_distant_registry_match_does_not_confirm(self):
        """A weak match helps the score but must not upgrade the verdict.

        The corroboration radius grows with drift time, so a match can now sit
        200 km away after a fortnight. That is real evidence and worth points,
        but calling it "confirmed" would let distance-scaled coincidences
        masquerade as independent confirmation.
        """
        near = assess(**real_slick(), corroboration=CORROBORATION)
        far = assess(**real_slick(), corroboration=DISTANT_CORROBORATION)

        assert near.tier == "confirmed"
        assert far.tier != "confirmed"
        assert far.corroborated, "it is still corroborated, just not strongly"

    def test_a_weaker_registry_match_is_worth_fewer_points(self):
        """The bonus scales with match quality rather than being flat.

        Measured on a middling candidate: a strong slick saturates the score
        cap, which would hide the difference.
        """
        middling = dict(p_oil=0.60, wind_window_score=0.8, damping_ratio=0.55,
                        elongation=6.0, area_km2=2.0, morphology="linear")

        near = assess(**middling, corroboration=CORROBORATION)
        far = assess(**middling, corroboration=DISTANT_CORROBORATION)

        assert far.score < near.score

    def test_strong_physics_alone_is_probable(self):
        a = assess(**real_slick())
        assert a.tier == "probable"
        assert a.is_actual_oil
        assert not a.corroborated

    def test_marginal_evidence_does_not_reach_oil(self):
        a = assess(p_oil=0.55, wind_window_score=0.3, damping_ratio=0.65,
                   elongation=3.0, area_km2=1.0, morphology="unknown")
        assert not a.is_actual_oil

    def test_weak_evidence_is_insufficient(self):
        a = assess(p_oil=0.51, wind_window_score=0.15, damping_ratio=0.85,
                   elongation=1.2, area_km2=0.15, morphology="blob")
        assert a.tier == "insufficient"

    def test_a_physics_rejection_settles_it(self):
        """Registry corroboration must not resurrect a patch the physics rejected."""
        a = assess(**real_slick(),
                   rejected_reason="wind 1.2 m/s below threshold",
                   corroboration=CORROBORATION)
        assert a.tier == "insufficient"
        assert a.score == 0.0
        assert not a.is_actual_oil

    def test_corroboration_cannot_carry_a_candidate_alone(self):
        """It is a strong signal, not a substitute for physical evidence."""
        a = assess(p_oil=0.51, wind_window_score=0.1, damping_ratio=0.9,
                   elongation=1.1, area_km2=0.1, morphology="blob",
                   corroboration=CORROBORATION)
        assert a.tier != "confirmed"

    def test_known_fixed_source_is_still_real_oil(self):
        """A persistent leak is a genuine spill, just never a vessel discharge."""
        a = assess(p_oil=0.62, wind_window_score=0.7, damping_ratio=0.4,
                   elongation=4.0, area_km2=8.0, morphology="blob",
                   source_type="infrastructure")
        assert a.is_actual_oil


class TestOrderingAndExplanations:
    def test_tier_order_is_monotonic(self):
        assert (TIER_ORDER["confirmed"] > TIER_ORDER["probable"]
                > TIER_ORDER["possible"] > TIER_ORDER["insufficient"])

    def test_score_rises_with_evidence(self):
        weak = assess(p_oil=0.55, wind_window_score=0.3, damping_ratio=0.7,
                      elongation=2.0, area_km2=0.5, morphology="unknown")
        strong = assess(**real_slick())
        assert strong.score > weak.score

    def test_every_assessment_explains_itself(self):
        a = assess(**real_slick())
        assert a.reasons
        assert all(len(r) > 8 for r in a.reasons)

    def test_serialises_with_its_meaning(self):
        d = assess(**real_slick()).to_dict()
        assert d["tier"] and d["meaning"] and "is_actual_oil" in d

    def test_missing_damping_is_handled(self):
        a = assess(**real_slick(damping_ratio=None))
        assert a.score >= 0.0  # must not raise
