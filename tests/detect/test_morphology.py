"""Morphology routing.

CLAUDE.md hard rule 4: exclude natural seeps and fixed infrastructure before
vessel attribution. Accusing a ship of a natural seep is the worst failure
this system can produce, so these tests guard that boundary.
"""

from __future__ import annotations

import pytest

from detect.morphology import KNOWN_SOURCES, classify_morphology, match_known_source
from detect.polygonize import RegionFeatures


def region(lon=72.0, lat=15.0, elongation=22.0, compactness=0.08):
    return RegionFeatures(
        label=1, pixel_count=900, area_km2=8.0, centroid_rc=(0.0, 0.0),
        centroid_lonlat=(lon, lat), elongation=elongation,
        compactness=compactness, orientation_deg=90.0, major_axis_km=12.0,
        minor_axis_km=0.6, damping_ratio=0.15, mean_db=-25.0, surround_db=-17.0,
        texture_homogeneity=0.7, texture_contrast=6.0, texture_variance=1.0,
        vh_vv_ratio=0.12, mean_confidence=0.9,
    )


class TestVesselRouting:
    def test_long_thin_streak_routes_to_vessel(self):
        verdict = classify_morphology(region(elongation=22.0, compactness=0.08))
        assert verdict.morphology == "linear"
        assert verdict.source_type == "vessel"
        assert verdict.goes_to_vessel_attribution

    def test_round_blob_does_not_route_to_vessel(self):
        verdict = classify_morphology(region(elongation=1.3, compactness=0.85))
        assert verdict.morphology == "blob"
        assert not verdict.goes_to_vessel_attribution

    def test_ambiguous_shape_is_not_attributed(self):
        verdict = classify_morphology(region(elongation=4.0, compactness=0.40))
        assert verdict.morphology == "unknown"
        assert not verdict.goes_to_vessel_attribution


class TestFixedSourceExclusion:
    def test_taylor_energy_is_never_blamed_on_a_ship(self):
        """A permanent leak site, however streak-like it looks that day."""
        verdict = classify_morphology(region(lon=-88.970, lat=28.936, elongation=25.0))
        assert verdict.source_type == "infrastructure"
        assert not verdict.goes_to_vessel_attribution
        assert "Taylor Energy" in verdict.reason

    def test_natural_seep_is_never_blamed_on_a_ship(self):
        verdict = classify_morphology(region(lon=-119.883, lat=34.391, elongation=30.0))
        assert verdict.source_type == "natural_seep"
        assert not verdict.goes_to_vessel_attribution

    def test_known_source_beats_a_linear_shape(self):
        """The catalogue check runs first and wins outright."""
        verdict = classify_morphology(
            region(lon=76.136, lat=9.3125, elongation=40.0, compactness=0.05)
        )
        assert verdict.source_type == "infrastructure"
        assert verdict.matched_source.name == "MSC ELSA 3 wreck"

    def test_open_ocean_matches_no_catalogue_entry(self):
        assert match_known_source(60.0, 5.0) is None

    def test_recurring_slick_is_treated_as_fixed(self):
        verdict = classify_morphology(region(elongation=20.0), seen_in_previous_passes=True)
        assert verdict.source_type == "infrastructure"
        assert not verdict.goes_to_vessel_attribution

    def test_every_catalogue_entry_is_well_formed(self):
        for source in KNOWN_SOURCES:
            assert -180 <= source.lon <= 180
            assert -90 <= source.lat <= 90
            assert source.kind in ("natural_seep", "infrastructure")
            assert source.note
