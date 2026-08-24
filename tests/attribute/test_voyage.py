"""Voyage reconstruction: where a vessel came from and where it went."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from attribute.ais import AISPosition, VesselTrack
from attribute.voyage import build_voyage, nearest_port, project_arrival
from core.geo import destination_point

T0 = datetime(2025, 5, 25, 0, 0, tzinfo=timezone.utc)
COLOMBO = (79.842, 6.951)
MUMBAI = (72.842, 18.944)


def track_between(start, end, mmsi="419000001", name="MV TEST",
                  speed_kn=12.0, step_min=20, t0=T0):
    from core.geo import bearing_deg, haversine_km

    course = bearing_deg(start, end)
    leg = speed_kn * 1.852 * step_min / 60.0
    steps = max(2, int(haversine_km(start, end) / leg))
    positions, pt, t = [], start, t0
    for _ in range(steps):
        p = AISPosition(mmsi, t, pt[0], pt[1], sog=speed_kn, cog=course, name=name)
        positions.append(p)
        pt = destination_point(pt, course, leg)
        t += timedelta(minutes=step_min)
    return VesselTrack(mmsi=mmsi, positions=positions, source="test")


class TestPorts:
    def test_finds_the_nearest_port(self):
        name, distance = nearest_port(*COLOMBO)
        assert "Colombo" in name
        assert distance < 5.0

    def test_mid_ocean_has_no_nearby_port(self):
        assert nearest_port(65.0, 0.0, max_km=200.0) is None


class TestVoyage:
    def test_reports_both_endpoints_with_times(self):
        voyage = build_voyage(track_between(COLOMBO, MUMBAI))
        data = voyage.to_dict()
        assert "Colombo" in data["from"]["nearest_port"]
        assert data["from"]["at"] < data["to"]["at"]
        assert data["distance_km"] > 1000

    def test_speed_and_duration_are_consistent(self):
        voyage = build_voyage(track_between(COLOMBO, MUMBAI, speed_kn=12.0))
        assert voyage.mean_speed_knots == pytest.approx(12.0, abs=0.6)
        implied = voyage.distance_km / 1.852 / voyage.duration_hours
        assert implied == pytest.approx(voyage.mean_speed_knots, abs=0.6)

    def test_declared_destination_is_carried_through(self):
        voyage = build_voyage(track_between(COLOMBO, MUMBAI), declared_destination="MUMBAI")
        assert voyage.to_dict()["declared_destination"] == "MUMBAI"

    def test_single_position_yields_no_voyage(self):
        one = VesselTrack("419", [AISPosition("419", T0, 76.0, 9.0)], source="t")
        assert build_voyage(one) is None

    def test_states_that_the_window_is_not_the_whole_voyage(self):
        """The endpoints are the limits of our AIS query, not of the passage."""
        note = build_voyage(track_between(COLOMBO, MUMBAI)).coverage_note
        assert "not of the" in note and "whole voyage" in note


class TestProjectedArrival:
    def test_projects_along_the_observed_course(self):
        from core.geo import bearing_deg

        course = bearing_deg(COLOMBO, MUMBAI)
        result = project_arrival(COLOMBO[0], COLOMBO[1], course, 12.0)
        assert result is not None
        assert result["basis"].startswith("projected")

    def test_labels_itself_a_projection_not_a_destination(self):
        """It must never be confused with what the vessel actually declared."""
        result = project_arrival(76.0, 10.0, 0.0, 12.0)
        assert result is not None
        assert "not a declared destination" in result["basis"] or "projection only" in result["basis"]

    def test_a_stationary_vessel_has_no_projection(self):
        assert project_arrival(76.0, 10.0, 90.0, 0.1) is None

    def test_unknown_course_has_no_projection(self):
        assert project_arrival(76.0, 10.0, None, 12.0) is None
