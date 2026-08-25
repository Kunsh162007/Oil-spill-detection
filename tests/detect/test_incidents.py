"""Documented incident registry and corroboration.

These are CONFIRMED spills, categorically different from our detections, and
the tests guard that distinction as much as the mechanics.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from detect.incidents import (
    IncidentRegistry,
    SpillIncident,
    load_registry,
    load_world_incidents,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NOAA_CSV = REPO_ROOT / "data" / "reference" / "noaa_incidents.csv"


def incident(name="Test spill", lon=76.0, lat=9.0, days_ago=0,
             commodity="crude oil", persistent=False):
    return SpillIncident(
        incident_id="test-1", name=name, lon=lon, lat=lat,
        occurred_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        commodity=commodity, source="test",
        extras={"persistent": persistent},
    )


class TestPetroleumFilter:
    """The map is about oil. Chemical and vegetable spills are not oil."""

    @pytest.mark.parametrize("commodity", [
        "crude oil", "Diesel", "bunker fuel", "heavy fuel oil",
        "condensate", "gasoline", "bilge water",
    ])
    def test_petroleum_recognised(self, commodity):
        assert incident(commodity=commodity).is_petroleum

    @pytest.mark.parametrize("commodity", [
        "vegetable oil", "palm oil", "molasses", "sulfuric acid",
        "ammonia", "fertilizer",
    ])
    def test_non_petroleum_excluded(self, commodity):
        assert not incident(commodity=commodity).is_petroleum

    def test_volume_converts_gallons_to_cubic_metres(self):
        i = incident()
        i.max_release_gallons = 1000.0
        assert i.volume_m3 == pytest.approx(3.785, abs=0.01)


class TestWorldCatalogue:
    def test_covers_every_ocean_basin(self):
        """The point of the catalogue is coverage NOAA's US focus lacks."""
        world = load_world_incidents()
        lons = [i.lon for i in world]
        assert min(lons) < -80 and max(lons) > 100
        assert any(60 < i.lon < 100 for i in world), "no Indian Ocean entries"
        assert any(-20 < i.lon < 50 and i.lat < 40 for i in world), "no Africa/Europe entries"

    def test_entries_are_well_formed(self):
        for i in load_world_incidents():
            assert -180 <= i.lon <= 180
            assert -90 <= i.lat <= 90
            assert i.name and i.source

    def test_includes_our_validation_case(self):
        names = {i.name for i in load_world_incidents()}
        assert any("ELSA 3" in n for n in names)

    def test_natural_seeps_are_flagged(self):
        seeps = [i for i in load_world_incidents() if i.extras.get("natural_seep")]
        assert seeps, "natural seeps must be catalogued so they are never blamed on a ship"


class TestCorroboration:
    @pytest.fixture
    def registry(self):
        return IncidentRegistry([
            incident(name="Nearby spill", lon=76.0, lat=9.0, days_ago=2),
            incident(name="Old spill", lon=76.0, lat=9.0, days_ago=400),
            incident(name="Far spill", lon=120.0, lat=9.0, days_ago=1),
        ])

    def test_matches_a_spill_at_the_same_place_and_time(self, registry):
        matches = registry.find(76.02, 9.02, datetime.now(timezone.utc))
        assert matches
        assert matches[0].incident.name == "Nearby spill"
        assert matches[0].confidence > 0.5

    def test_open_ocean_matches_nothing(self, registry):
        assert registry.find(0.0, -40.0, datetime.now(timezone.utc)) == []

    def test_a_detection_before_the_incident_is_not_that_incident(self, registry):
        """Oil cannot be detected before it was spilled."""
        earlier = datetime.now(timezone.utc) - timedelta(days=30)
        assert registry.find(76.0, 9.0, earlier) == []

    def test_a_long_stale_incident_does_not_corroborate(self, registry):
        """A slick disperses; a spill 400 days ago cannot explain today's patch."""
        names = [m.incident.name for m in registry.find(76.0, 9.0, datetime.now(timezone.utc))]
        assert "Old spill" not in names

    def test_persistent_sources_match_at_any_time(self):
        registry = IncidentRegistry([
            incident(name="Taylor Energy", lon=-88.97, lat=28.94, days_ago=8000, persistent=True)
        ])
        matches = registry.find(-88.97, 28.94, datetime.now(timezone.utc))
        assert matches and matches[0].confidence > 0.8

    def test_reason_is_human_readable(self, registry):
        reason = registry.find(76.02, 9.02, datetime.now(timezone.utc))[0].reason
        assert "km" in reason and len(reason) > 25


@pytest.mark.skipif(not NOAA_CSV.exists(), reason="NOAA registry not downloaded")
class TestRealRegistry:
    def test_loads_thousands_of_real_incidents(self):
        registry = load_registry(NOAA_CSV)
        assert len(registry) > 1000

    def test_coverage_is_global(self):
        registry = load_registry(NOAA_CSV)
        lons = [i.lon for i in registry.incidents]
        assert min(lons) < -150 and max(lons) > 150

    def test_longitudes_are_wrapped_into_range(self):
        """The raw feed carries a few unwrapped values such as 237.4."""
        for i in load_registry(NOAA_CSV).incidents:
            assert -180 <= i.lon <= 180
            assert -90 <= i.lat <= 90

    def test_deepwater_horizon_is_present(self):
        registry = load_registry(NOAA_CSV)
        matches = registry.find(-88.366, 28.738, datetime(2010, 4, 25, tzinfo=timezone.utc))
        assert matches, "the largest marine spill on record should corroborate"


class TestDriftScaledRadius:
    """Corroboration distance has to grow with elapsed time.

    Oil moves. Demanding that a slick found a fortnight after an incident still
    sit within a fixed 60 km asks it to have stayed put, which is the one thing
    oil never does - and it is why real MSC ELSA 3 detections 15 days later, 87
    km away, were scored as uncorroborated.
    """

    def test_radius_grows_with_elapsed_days(self):
        from detect.incidents import (CORROBORATION_BASE_KM,
                                      CORROBORATION_MAX_KM,
                                      corroboration_radius_km)

        same_day = corroboration_radius_km(0.0)
        a_week = corroboration_radius_km(7.0)
        a_fortnight = corroboration_radius_km(14.0)

        assert same_day == CORROBORATION_BASE_KM
        assert a_week > same_day
        assert a_fortnight > a_week
        assert corroboration_radius_km(400.0) == CORROBORATION_MAX_KM

    def test_unknown_or_negative_elapsed_time_uses_the_base(self):
        """No elapsed time means no drift allowance - never a wider net."""
        from detect.incidents import (CORROBORATION_BASE_KM,
                                      corroboration_radius_km)

        assert corroboration_radius_km(None) == CORROBORATION_BASE_KM
        assert corroboration_radius_km(-3.0) == CORROBORATION_BASE_KM

    def test_a_leaking_wreck_still_corroborates_mid_episode(self):
        """MSC ELSA 3: the regression this whole change exists to fix."""
        from datetime import datetime, timezone

        from detect.incidents import load_registry

        registry = load_registry(None)          # curated catalogue only
        when = datetime(2025, 6, 9, 0, 41, tzinfo=timezone.utc)

        matches = registry.find(75.95, 8.55, when)          # ~87 km south

        assert matches, "a detection 15 days later and 87 km away must match"
        assert "ELSA 3" in matches[0].incident.name
        assert 0.0 < matches[0].confidence <= 1.0
