"""Forward drift - where is the slick now, and when to refuse to say."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.geo import haversine_km
from drift.forward import (
    FORECAST_RELIABLE_HOURS,
    MAX_FORECAST_HOURS,
    forward_drift,
)
from drift.readers import ConstantField, WindField, zero_field

CENTRE = (76.30, 9.31)
D = 0.02
POLY = [
    (CENTRE[0] - D, CENTRE[1] - D), (CENTRE[0] + D, CENTRE[1] - D),
    (CENTRE[0] + D, CENTRE[1] + D), (CENTRE[0] - D, CENTRE[1] + D),
    (CENTRE[0] - D, CENTRE[1] - D),
]


def observed(hours_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


class TestAdvection:
    def test_displacement_matches_hand_calculation(self):
        """0.5 m/s east for 12 h = 21.6 km."""
        result = forward_drift(
            POLY, observed(12), None,
            ConstantField.from_speed_direction(0.5, 90.0),
            wind=zero_field(), n_particles=150, diffusion_m2_s=0.0, seed=1,
        )
        assert result.stats["displacement_km"] == pytest.approx(21.6, abs=0.6)

    def test_moves_downstream(self):
        result = forward_drift(
            POLY, observed(10), None,
            ConstantField.from_speed_direction(0.5, 90.0),
            n_particles=100, diffusion_m2_s=0.0, seed=1,
        )
        assert result.state.lon > CENTRE[0]  # flow is eastward

    def test_forward_undoes_backward(self):
        """The two directions must use the same physics, not two models."""
        from drift.backward import backtrack

        currents = ConstantField.from_speed_direction(0.4, 45.0)
        obs = datetime(2025, 5, 25, 6, 0, tzinfo=timezone.utc)
        back = backtrack(POLY, obs, currents, wind=zero_field(),
                         backtrack_hours=10.0, n_particles=120,
                         diffusion_m2_s=0.0, seed=5)
        forward = forward_drift(
            POLY, obs, obs + timedelta(hours=10), currents, wind=zero_field(),
            n_particles=120, diffusion_m2_s=0.0, seed=5,
        )
        moved_back = haversine_km(CENTRE, (back.origin.lon, back.origin.lat))
        moved_fwd = haversine_km(CENTRE, (forward.state.lon, forward.state.lat))
        assert moved_back == pytest.approx(moved_fwd, rel=0.05)

    def test_wind_contributes_three_percent(self):
        currents = ConstantField.from_speed_direction(0.5, 90.0)
        wind = WindField(ConstantField.from_speed_direction(10.0, 90.0), 0.03)
        result = forward_drift(POLY, observed(12), None, currents, wind=wind,
                               n_particles=100, diffusion_m2_s=0.0, seed=1)
        assert result.stats["displacement_km"] == pytest.approx((0.5 + 0.3) * 12 * 3.6, abs=0.8)


class TestHonestLimits:
    def test_refuses_to_forecast_a_stale_scene(self):
        """Advecting for weeks yields a confident position across an ocean.

        Real oil disperses, strands or weathers long before that, so the
        honest answer is to decline rather than extrapolate.
        """
        result = forward_drift(
            POLY, observed(MAX_FORECAST_HOURS + 500), None,
            ConstantField.from_speed_direction(0.5, 90.0), n_particles=40,
        )
        assert result.state.label == "historical"
        assert not result.reliable
        assert result.track == []
        assert haversine_km(CENTRE, (result.state.lon, result.state.lat)) < 5.0

    def test_long_but_allowed_horizon_is_flagged_unreliable(self):
        result = forward_drift(
            POLY, observed(FORECAST_RELIABLE_HOURS + 10), None,
            ConstantField.from_speed_direction(0.3, 90.0), n_particles=40,
        )
        assert not result.reliable
        assert any("search area" in w for w in result.warnings)

    def test_always_states_that_weathering_is_not_modelled(self):
        result = forward_drift(POLY, observed(6), None,
                               ConstantField.from_speed_direction(0.3, 90.0),
                               n_particles=40)
        assert any("weathering" in w for w in result.warnings)

    def test_uncertainty_grows_with_time(self):
        currents = ConstantField.from_speed_direction(0.3, 90.0)
        short = forward_drift(POLY, observed(3), None, currents, n_particles=80, seed=2)
        long = forward_drift(POLY, observed(36), None, currents, n_particles=80, seed=2)
        assert long.state.uncertainty_km > short.state.uncertainty_km

    def test_future_observation_is_not_advected(self):
        result = forward_drift(POLY, observed(-5), None,
                               ConstantField.from_speed_direction(0.5, 90.0),
                               n_particles=40)
        assert result.track == []


class TestFootprint:
    def test_reports_a_drifted_footprint(self):
        result = forward_drift(POLY, observed(6), None,
                               ConstantField.from_speed_direction(0.3, 90.0),
                               n_particles=120, seed=3)
        assert len(result.state.polygon) >= 4
        for lon, lat in result.state.polygon:
            assert -180 <= lon <= 180 and -90 <= lat <= 90

    def test_serialises_cleanly(self):
        d = forward_drift(POLY, observed(6), None,
                          ConstantField.from_speed_direction(0.3, 90.0),
                          n_particles=60).state.to_dict()
        assert d["label"] and d["description"] and "uncertainty_km" in d
