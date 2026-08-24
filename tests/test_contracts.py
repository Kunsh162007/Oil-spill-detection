"""Contract invariants. These are frozen; breaking one breaks every module."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest

from core.contracts import (
    WIND_HIGH_CUT_MS,
    WIND_LOW_CUT_MS,
    DriftOrigin,
    WindContext,
    to_dict,
    to_json,
    wind_window_score,
)


class TestWindWindow:
    def test_calm_water_scores_zero(self):
        """Below the floor, calm sea is indistinguishable from oil."""
        assert wind_window_score(0.0) == 0.0
        assert wind_window_score(1.2) == 0.0
        assert wind_window_score(WIND_LOW_CUT_MS) == 0.0

    def test_storm_scores_zero(self):
        """Above the ceiling, oil mixes into the wave field."""
        assert wind_window_score(WIND_HIGH_CUT_MS) == 0.0
        assert wind_window_score(20.0) == 0.0

    def test_mid_window_scores_one(self):
        for speed in (4.0, 5.5, 6.5, 7.5):
            assert wind_window_score(speed) == 1.0

    def test_graded_not_binary(self):
        """The window has soft edges: published bounds disagree by m/s."""
        marginal = wind_window_score(2.6)
        assert 0.0 < marginal < 1.0

    def test_monotonic_on_each_flank(self):
        rising = [wind_window_score(s) for s in (2.1, 2.5, 3.0, 3.4)]
        assert rising == sorted(rising)
        falling = [wind_window_score(s) for s in (8.0, 9.0, 10.5, 11.9)]
        assert falling == sorted(falling, reverse=True)

    def test_nan_is_not_a_window(self):
        assert wind_window_score(float("nan")) == 0.0

    def test_from_speed_populates_score(self):
        ctx = WindContext.from_speed(5.5, 180.0)
        assert ctx.window_score == 1.0
        assert ctx.source == "ERA5"


class TestSerialisation:
    def test_nan_never_reaches_json(self):
        """NaN is not valid JSON and breaks strict parsers, browsers included."""
        origin = DriftOrigin(
            lon=float("nan"), lat=9.3,
            estimated_at=datetime(2025, 5, 25, tzinfo=timezone.utc),
            uncertainty_km=float("inf"), n_particles=100,
        )
        payload = to_dict(origin)
        assert payload["lon"] is None
        assert payload["uncertainty_km"] is None
        json.loads(to_json(origin))  # must not raise

    def test_datetimes_serialise_as_utc_iso(self):
        origin = DriftOrigin(
            lon=76.1, lat=9.3,
            estimated_at=datetime(2025, 5, 25, 5, 42, tzinfo=timezone.utc),
            uncertainty_km=8.0, n_particles=100,
        )
        assert to_dict(origin)["estimated_at"] == "2025-05-25T05:42:00+00:00"


class TestDriftReliability:
    def test_long_backtrack_is_unreliable(self):
        """Beyond ~24 h the origin is a wide blur, not a point."""
        origin = DriftOrigin(
            lon=76.1, lat=9.3, estimated_at=datetime.now(timezone.utc),
            uncertainty_km=10.0, n_particles=500, backtrack_hours=30.0,
        )
        assert not origin.is_reliable

    def test_wide_uncertainty_is_unreliable(self):
        origin = DriftOrigin(
            lon=76.1, lat=9.3, estimated_at=datetime.now(timezone.utc),
            uncertainty_km=90.0, n_particles=500, backtrack_hours=6.0,
        )
        assert not origin.is_reliable

    def test_short_tight_run_is_reliable(self):
        origin = DriftOrigin(
            lon=76.1, lat=9.3, estimated_at=datetime.now(timezone.utc),
            uncertainty_km=8.0, n_particles=500, backtrack_hours=12.0,
        )
        assert origin.is_reliable
