"""Vessel ranking. Correlation, never proof - and never a wrong accusation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from attribute.ais import AISPosition, VesselTrack
from attribute.scoring import (
    ScoringContext,
    parity_score,
    proximity_score,
    rank_vessels,
    temporality_score,
)
from core.contracts import DriftOrigin
from core.geo import destination_point

RELEASE = datetime(2025, 5, 25, 2, 0, tzinfo=timezone.utc)
ORIGIN_PT = (76.10, 9.31)


@pytest.fixture
def origin():
    return DriftOrigin(
        lon=ORIGIN_PT[0], lat=ORIGIN_PT[1], estimated_at=RELEASE,
        uncertainty_km=8.0, n_particles=500, backtrack_hours=12.0,
        method="analytical-advection",
    )


def make_track(mmsi, name, start, course, n=12, step_min=20, t0=None,
               speed_kn=12.0, gap_after=None):
    t0 = t0 or (RELEASE - timedelta(hours=2))
    positions, pt, t = [], start, t0
    for i in range(n):
        if gap_after is not None and i == gap_after:
            t += timedelta(hours=3)
        positions.append(AISPosition(mmsi, t, pt[0], pt[1], sog=speed_kn,
                                     cog=course, name=name))
        pt = destination_point(pt, course, speed_kn * 1.852 * step_min / 60.0)
        t += timedelta(minutes=step_min)
    return VesselTrack(mmsi=mmsi, positions=positions, source="test")


class TestParity:
    def test_parallel_course_scores_high(self):
        track = make_track("1", "A", destination_point(ORIGIN_PT, 270, 15), 90)
        score, note = parity_score(track, 90.0)
        assert score > 0.95
        assert "deg" in note

    def test_reciprocal_course_is_equally_parallel(self):
        """A slick axis is undirected: 090 and 270 both lie along an E-W streak.

        Comparing bearings instead of axes here would be a 90 degree error and
        would rank the wrong ship first.
        """
        east = make_track("1", "A", destination_point(ORIGIN_PT, 270, 15), 90)
        west = make_track("2", "B", destination_point(ORIGIN_PT, 90, 15), 270)
        assert parity_score(east, 90.0)[0] == pytest.approx(
            parity_score(west, 90.0)[0], abs=0.02
        )

    def test_perpendicular_course_scores_zero(self):
        track = make_track("3", "C", destination_point(ORIGIN_PT, 180, 12), 0)
        assert parity_score(track, 90.0)[0] < 0.05

    def test_blob_has_no_axis_to_compare(self):
        track = make_track("4", "D", ORIGIN_PT, 90)
        score, note = parity_score(track, None)
        assert score == 0.0
        assert "blob" in note.lower() or "no slick axis" in note.lower()


class TestProximity:
    def test_passing_through_origin_scores_high(self, origin):
        track = make_track("1", "A", destination_point(ORIGIN_PT, 270, 15), 90)
        score, _, dist, _ = proximity_score(track, origin, 50.0)
        assert score > 0.9
        assert dist < 2.0

    def test_beyond_search_radius_scores_zero(self, origin):
        track = make_track("2", "B", destination_point(ORIGIN_PT, 90, 200), 90)
        score, note, _, _ = proximity_score(track, origin, 50.0)
        assert score == 0.0
        assert "beyond" in note

    def test_tolerance_scales_with_drift_uncertainty(self, origin):
        """A 15 km approach is close when the origin is a 20 km blur."""
        track = make_track("3", "C", destination_point(ORIGIN_PT, 270, 30), 90, n=4)
        tight = proximity_score(track, origin, 60.0)[0]
        origin.uncertainty_km = 25.0
        loose = proximity_score(track, origin, 60.0)[0]
        assert loose > tight


class TestTemporality:
    def test_passing_before_release_scores_well(self):
        score, note = temporality_score(RELEASE - timedelta(hours=1), RELEASE, 8.0)
        assert score > 0.8
        assert "before" in note

    def test_arriving_after_decays_faster(self):
        """A ship arriving after the oil was already there could not have done it."""
        before = temporality_score(RELEASE - timedelta(hours=3), RELEASE, 8.0)[0]
        after = temporality_score(RELEASE + timedelta(hours=3), RELEASE, 8.0)[0]
        assert after < before / 2

    def test_missing_timestamp_scores_zero(self):
        assert temporality_score(None, RELEASE, 8.0)[0] == 0.0


class TestRanking:
    def test_guilty_vessel_ranks_first(self, origin):
        upstream = destination_point(ORIGIN_PT, 270, 15)
        tracks = [
            make_track("111", "SUSPECT", upstream, 90),
            make_track("333", "CROSSING", destination_point(ORIGIN_PT, 180, 12), 0),
        ]
        ctx = ScoringContext(origin=origin, slick_axis_deg=90.0, release_time=RELEASE)
        ranked = rank_vessels(tracks, ctx, top_n=3)
        assert ranked[0].name == "SUSPECT"

    def test_dark_vessel_outranks_an_equal_honest_one(self, origin):
        """Going silent exactly where a slick starts is a stronger signal."""
        upstream = destination_point(ORIGIN_PT, 270, 15)
        honest = make_track("111", "HONEST", upstream, 90)
        dark = make_track("222", "DARK", upstream, 90, gap_after=4)
        ctx = ScoringContext(origin=origin, slick_axis_deg=90.0, release_time=RELEASE)
        ranked = {c.name: c for c in rank_vessels([honest, dark], ctx, top_n=5)}
        assert ranked["DARK"].went_dark is True
        assert "WENT DARK" in ranked["DARK"].evidence

    def test_distant_vessel_is_excluded_despite_perfect_parity(self, origin):
        """A necessary condition: it must actually have been near the origin."""
        far = make_track("444", "FAR", destination_point(ORIGIN_PT, 90, 150), 90)
        ctx = ScoringContext(origin=origin, slick_axis_deg=90.0, release_time=RELEASE)
        assert [c.name for c in rank_vessels([far], ctx, top_n=3)] == []

    def test_evidence_is_human_readable(self, origin):
        track = make_track("111", "SUSPECT", destination_point(ORIGIN_PT, 270, 15), 90)
        ctx = ScoringContext(origin=origin, slick_axis_deg=90.0, release_time=RELEASE)
        evidence = rank_vessels([track], ctx, top_n=1)[0].evidence
        assert "km from the estimated origin" in evidence
        assert len(evidence) > 40
