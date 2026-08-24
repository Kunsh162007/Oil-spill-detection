"""Abstention. CLAUDE.md hard rule 5: abstain when uncertain.

A confident wrong accusation costs far more than an honest shrug, and a
system that never abstains is not measuring anything.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.contracts import DriftOrigin, SlickCandidate, VesselCandidate, WindContext
from decision.rank import DecisionConfig, coverage_risk, decide

NOW = datetime(2025, 5, 25, 5, 42, tzinfo=timezone.utc)
CFG = DecisionConfig()


def slick(wind_ms=5.5, rejected=None):
    return SlickCandidate(
        candidate_id="TEST-001", scene_id="TEST", polygon_wkt="POLYGON EMPTY",
        area_km2=8.0, elongation=20.0, compactness=0.1, damping_ratio=0.15,
        wind=WindContext.from_speed(wind_ms, 200.0), p_oil=0.9,
        rejected_reason=rejected, morphology="linear",
    )


def origin(uncertainty_km=8.0):
    return DriftOrigin(lon=76.1, lat=9.3, estimated_at=NOW,
                       uncertainty_km=uncertainty_km, n_particles=500,
                       backtrack_hours=12.0, method="analytical-advection")


def vessel(name, score):
    return VesselCandidate(
        mmsi="419000001", name=name, vessel_type="Cargo", flag="IN",
        parity=0.9, proximity=0.9, temporality=0.9, score=score,
        went_dark=False, evidence="test evidence for this candidate vessel",
    )


class TestAbstains:
    def test_when_the_top_two_are_within_noise(self):
        result = decide(slick(), origin(), "vessel",
                        [vessel("A", 0.82), vessel("B", 0.79)], CFG)
        assert result.abstained
        assert "too close to separate" in result.abstain_reason

    def test_when_no_vessel_fits_well(self):
        result = decide(slick(), origin(), "vessel", [vessel("A", 0.2)], CFG)
        assert result.abstained
        assert "below the" in result.abstain_reason

    def test_when_no_ais_coverage(self):
        result = decide(slick(), origin(), "vessel", [], CFG)
        assert result.abstained
        assert "AIS" in result.abstain_reason

    def test_when_wind_is_outside_the_detection_window(self):
        """Detection itself is unreliable there, so attribution is withheld."""
        result = decide(slick(wind_ms=1.2), origin(), "vessel", [vessel("A", 0.9)], CFG)
        assert result.abstained
        assert "wind" in result.abstain_reason.lower()

    def test_when_drift_origin_is_too_uncertain(self):
        result = decide(slick(), origin(uncertainty_km=120.0), "vessel",
                        [vessel("A", 0.9)], CFG)
        assert result.abstained
        assert "uncertain" in result.abstain_reason

    def test_when_there_is_no_origin_at_all(self):
        assert decide(slick(), None, "vessel", [vessel("A", 0.9)], CFG).abstained

    def test_when_the_slick_was_rejected_as_a_lookalike(self):
        result = decide(slick(rejected="wind 1.2 m/s below threshold"), origin(),
                        "vessel", [vessel("A", 0.95)], CFG)
        assert result.abstained
        assert "not classified as oil" in result.abstain_reason

    def test_for_a_natural_seep(self):
        """Never attribute a seep to a vessel."""
        result = decide(slick(), origin(), "natural_seep", [vessel("A", 0.95)], CFG)
        assert result.abstained
        assert "natural seep" in result.abstain_reason


class TestRanks:
    def test_a_clear_winner_is_returned(self):
        result = decide(slick(), origin(), "vessel",
                        [vessel("A", 0.91), vessel("B", 0.42)], CFG)
        assert not result.abstained
        assert result.candidates[0].name == "A"
        assert result.evidence["margin"] > CFG.abstain_margin


class TestEvidenceBundle:
    def test_carries_wind_drift_and_the_disclaimer(self):
        result = decide(slick(), origin(), "vessel",
                        [vessel("A", 0.91), vessel("B", 0.4)], CFG)
        assert result.evidence["wind"]["speed_ms"] == 5.5
        assert result.evidence["drift"]["uncertainty_km"] == 8.0
        assert "not evidence" in result.evidence["disclaimer"]

    def test_candidates_are_kept_even_when_abstaining(self):
        """The reader should see what was considered, marked inconclusive."""
        result = decide(slick(), origin(), "vessel",
                        [vessel("A", 0.82), vessel("B", 0.80)], CFG)
        assert result.abstained
        assert len(result.candidates) == 2


class TestCoverageRisk:
    def test_reports_the_coverage_curve(self):
        good = decide(slick(), origin(), "vessel",
                      [vessel("A", 0.9), vessel("B", 0.3)], CFG)
        bad = decide(slick(), origin(), "vessel", [], CFG)
        stats = coverage_risk([good, bad, bad])
        assert stats["total"] == 3
        assert stats["abstained"] == 2
        assert stats["coverage"] == pytest.approx(1 / 3, abs=1e-4)  # stored rounded to 4dp
        assert stats["reasons"]

    def test_handles_an_empty_run(self):
        assert coverage_risk([])["coverage"] == 0.0
