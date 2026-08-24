"""Backward drift. Verified against displacements we can compute by hand."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.geo import haversine_km
from drift.backward import backtrack, seed_particles
from drift.readers import ConstantField, SyntheticField, WindField, zero_field

OBSERVED_AT = datetime(2025, 5, 25, 6, 0, tzinfo=timezone.utc)
CENTRE = (76.30, 9.31)
POLY = [
    (CENTRE[0] - 0.02, CENTRE[1] - 0.02), (CENTRE[0] + 0.02, CENTRE[1] - 0.02),
    (CENTRE[0] + 0.02, CENTRE[1] + 0.02), (CENTRE[0] - 0.02, CENTRE[1] + 0.02),
    (CENTRE[0] - 0.02, CENTRE[1] - 0.02),
]


class TestSeeding:
    def test_particles_land_inside_the_polygon(self):
        import numpy as np
        from shapely.geometry import Point, Polygon

        pts = seed_particles(POLY, 200, np.random.default_rng(0))
        poly = Polygon(POLY)
        assert len(pts) == 200
        assert all(poly.contains(Point(p)) for p in pts)

    def test_degenerate_polygon_raises(self):
        import numpy as np

        with pytest.raises(ValueError):
            seed_particles([(0, 0), (1, 1)], 10, np.random.default_rng(0))


class TestAdvectionPhysics:
    def test_displacement_matches_hand_calculation(self):
        """0.5 m/s east for 12 h = 21.6 km; backtracking must undo exactly that."""
        currents = ConstantField.from_speed_direction(0.5, 90.0, source="test")
        result = backtrack(POLY, OBSERVED_AT, currents, wind=zero_field(),
                           backtrack_hours=12.0, timestep_minutes=30.0,
                           n_particles=200, diffusion_m2_s=0.0, seed=42)
        expected_km = 0.5 * 12 * 3600 / 1000.0
        actual_km = haversine_km(CENTRE, (result.origin.lon, result.origin.lat))
        assert actual_km == pytest.approx(expected_km, abs=0.5)

    def test_origin_lies_upstream(self):
        currents = ConstantField.from_speed_direction(0.5, 90.0, source="test")
        result = backtrack(POLY, OBSERVED_AT, currents, wind=zero_field(),
                           backtrack_hours=12.0, n_particles=100,
                           diffusion_m2_s=0.0, seed=1)
        assert result.origin.lon < CENTRE[0]  # flow was eastward, so origin is west

    def test_origin_time_is_the_release_time(self):
        currents = ConstantField.from_speed_direction(0.3, 45.0)
        result = backtrack(POLY, OBSERVED_AT, currents, backtrack_hours=8.0,
                           n_particles=50, seed=0)
        assert result.origin.estimated_at == OBSERVED_AT - timedelta(hours=8)

    def test_wind_adds_three_percent_of_its_speed(self):
        """Standard surface-oil wind drift factor."""
        currents = ConstantField.from_speed_direction(0.5, 90.0)
        wind = WindField(ConstantField.from_speed_direction(10.0, 90.0), 0.03)
        result = backtrack(POLY, OBSERVED_AT, currents, wind=wind,
                           backtrack_hours=12.0, n_particles=100,
                           diffusion_m2_s=0.0, seed=42)
        expected = (0.5 + 0.3) * 12 * 3.6
        actual = haversine_km(CENTRE, (result.origin.lon, result.origin.lat))
        assert actual == pytest.approx(expected, abs=0.6)


class TestUncertainty:
    def test_diffusion_widens_uncertainty_without_moving_the_centre(self):
        currents = ConstantField.from_speed_direction(0.5, 90.0)
        tight = backtrack(POLY, OBSERVED_AT, currents, backtrack_hours=12.0,
                          n_particles=300, diffusion_m2_s=0.0, seed=42)
        spread = backtrack(POLY, OBSERVED_AT, currents, backtrack_hours=12.0,
                           n_particles=300, diffusion_m2_s=25.0, seed=42)
        assert spread.origin.uncertainty_km > tight.origin.uncertainty_km
        shift = haversine_km(
            (tight.origin.lon, tight.origin.lat),
            (spread.origin.lon, spread.origin.lat),
        )
        assert shift < 3.0

    def test_uncertainty_grows_with_backtrack_time(self):
        """Advection error accumulates; a longer rewind cannot be more precise."""
        currents = ConstantField.from_speed_direction(0.4, 120.0)
        short = backtrack(POLY, OBSERVED_AT, currents, backtrack_hours=4.0,
                          n_particles=150, seed=3)
        long = backtrack(POLY, OBSERVED_AT, currents, backtrack_hours=20.0,
                         n_particles=150, seed=3)
        assert long.origin.uncertainty_km > short.origin.uncertainty_km

    def test_long_backtrack_warns(self):
        currents = ConstantField.from_speed_direction(0.4, 120.0)
        result = backtrack(POLY, OBSERVED_AT, currents, backtrack_hours=30.0,
                           timestep_minutes=60.0, n_particles=60, seed=3)
        assert not result.origin.is_reliable
        assert any("24 h" in w for w in result.warnings)


class TestHonesty:
    def test_synthetic_currents_are_labelled_all_the_way_out(self):
        """CLAUDE.md rule 6: never pass invented data off as measurement."""
        result = backtrack(POLY, OBSERVED_AT, SyntheticField(), backtrack_hours=6.0,
                           n_particles=60, seed=0)
        assert "SYNTHETIC" in result.origin.method
        assert any("synthetic" in w.lower() for w in result.warnings)

    def test_track_runs_oldest_first(self):
        """The UI plays this forward, so the oil must travel the way it did."""
        currents = ConstantField.from_speed_direction(0.5, 90.0)
        result = backtrack(POLY, OBSERVED_AT, currents, backtrack_hours=6.0,
                           timestep_minutes=30.0, n_particles=50, seed=0)
        track = result.origin.track
        assert track[0]["hours_before"] > track[-1]["hours_before"]
        assert track[-1]["hours_before"] == 0.0
