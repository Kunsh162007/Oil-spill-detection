"""End to end, on the real synthetic scene.

The scene has a planted bilge dump plus three look-alike classes. These tests
assert the pipeline finds the first and rejects the others - which is the
whole claim the project makes.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


class TestDetection:
    def test_finds_the_planted_bilge_dump(self, demo_analysis):
        confirmed = demo_analysis.confirmed
        assert confirmed, "the planted linear slick was not detected"
        biggest = max(confirmed, key=lambda c: c.area_km2)
        assert biggest.morphology == "linear"
        assert biggest.elongation > 10.0
        assert biggest.p_oil > 0.7

    def test_rejects_the_look_alikes(self, demo_analysis):
        """Three look-alike classes were planted; all must be rejected."""
        assert len(demo_analysis.rejected) >= 3

    def test_exactly_one_slick_survives(self, demo_analysis):
        """One dump was planted. More than one survivor means look-alikes leaked."""
        assert len(demo_analysis.confirmed) == 1

    def test_every_rejection_states_a_reason(self, demo_analysis):
        for candidate in demo_analysis.rejected:
            assert candidate.rejected_reason
            assert len(candidate.rejected_reason) > 20

    def test_internal_waves_rejected_as_a_train(self, demo_analysis):
        """Judged alone every band passes; the periodicity is what betrays them."""
        reasons = [c.rejected_reason or "" for c in demo_analysis.rejected]
        assert any("internal-wave train" in r for r in reasons)

    def test_wind_context_present_on_every_candidate(self, demo_analysis):
        """CLAUDE.md rule 3: a candidate without wind context is not a candidate."""
        for candidate in demo_analysis.candidates:
            assert candidate.wind is not None
            assert candidate.wind.source


class TestAttribution:
    def test_confirmed_slick_gets_a_drift_origin(self, demo_analysis):
        slick = demo_analysis.confirmed[0]
        attribution = next(
            a for a in demo_analysis.attributions if a.candidate_id == slick.candidate_id
        )
        assert attribution.origin is not None
        assert attribution.origin.n_particles > 0

    def test_origin_predates_the_acquisition(self, demo_analysis):
        slick = demo_analysis.confirmed[0]
        attribution = next(
            a for a in demo_analysis.attributions if a.candidate_id == slick.candidate_id
        )
        assert attribution.origin.estimated_at < demo_analysis.scene.acquired_at

    def test_ranked_candidates_not_a_single_accusation(self, demo_analysis):
        """CLAUDE.md hard rule 1."""
        slick = demo_analysis.confirmed[0]
        attribution = next(
            a for a in demo_analysis.attributions if a.candidate_id == slick.candidate_id
        )
        assert len(attribution.candidates) > 1
        scores = [c.score for c in attribution.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_dark_vessel_is_identified(self, demo_analysis):
        slick = demo_analysis.confirmed[0]
        attribution = next(
            a for a in demo_analysis.attributions if a.candidate_id == slick.candidate_id
        )
        assert any(c.went_dark for c in attribution.candidates)

    def test_rejected_slicks_never_reach_vessel_attribution(self, demo_analysis):
        """Never accuse a ship over a look-alike."""
        rejected_ids = {c.candidate_id for c in demo_analysis.rejected}
        for attribution in demo_analysis.attributions:
            if attribution.candidate_id in rejected_ids:
                assert attribution.abstained
                assert not [c for c in attribution.candidates if c.score > 0]

    def test_every_attribution_carries_the_disclaimer(self, demo_analysis):
        for attribution in demo_analysis.attributions:
            assert "not evidence" in attribution.evidence["disclaimer"]


class TestPerformance:
    def test_scene_completes_inside_the_latency_budget(self, demo_analysis):
        """CLAUDE.md speed budget: a full scene end to end in under two minutes."""
        assert demo_analysis.stats["total_s"] < 120.0

    def test_timings_recorded_per_stage(self, demo_analysis):
        for stage in ("ingest", "stage_a", "stage_b", "polygonize", "lookalike"):
            assert stage in demo_analysis.timings
