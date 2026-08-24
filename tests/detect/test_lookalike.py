"""The look-alike physics stage - our differentiator.

Each test names a real look-alike class from CLAUDE.md's table and asserts
both the verdict AND that the printed reason is intelligible, because an
unexplainable rejection is worth nothing on stage.
"""

from __future__ import annotations

import pytest

from core.contracts import WindContext
from detect.lookalike import GateConfig, LookalikeModel
from detect.polygonize import RegionFeatures


def region(damping_db=8.0, elongation=22.0, compactness=0.08,
           homogeneity=0.78, contrast=6.0, area_km2=8.0, vh_vv=0.12):
    """A region with tunable physics. Defaults describe a classic bilge dump."""
    return RegionFeatures(
        label=1, pixel_count=1200, area_km2=area_km2,
        centroid_rc=(0.0, 0.0), centroid_lonlat=(72.0, 15.0),
        elongation=elongation, compactness=compactness,
        orientation_deg=90.0, major_axis_km=12.0, minor_axis_km=0.6,
        damping_ratio=10 ** (-damping_db / 10.0),
        mean_db=-25.0, surround_db=-17.0,
        texture_homogeneity=homogeneity, texture_contrast=contrast,
        texture_variance=1.0, vh_vv_ratio=vh_vv, mean_confidence=0.85,
    )


@pytest.fixture
def model():
    return LookalikeModel()


class TestWindGates:
    """CLAUDE.md rule 3: no detection without a wind check."""

    def test_calm_wind_rejects_even_a_perfect_slick(self, model):
        v = model.classify(region(), WindContext.from_speed(1.2, 190))
        assert not v.is_oil
        assert v.gate_hit == "wind_too_low"
        assert "1.2 m/s" in v.rejected_reason
        assert "calm water" in v.rejected_reason.lower()

    def test_storm_wind_rejects(self, model):
        v = model.classify(region(), WindContext.from_speed(14.0, 250))
        assert not v.is_oil
        assert v.gate_hit == "wind_too_high"
        assert "wave field" in v.rejected_reason

    def test_missing_wind_rejects(self, model):
        """A candidate without wind context is not a candidate."""
        v = model.classify(region(), WindContext.from_speed(float("nan")))
        assert not v.is_oil
        assert v.gate_hit == "no_wind"

    def test_gates_outrank_the_score(self, model):
        """No amount of learned confidence overrides physical impossibility."""
        strong = region(damping_db=10.0, elongation=40.0)
        v = model.classify(strong, WindContext.from_speed(0.5))
        assert not v.is_oil
        assert v.p_oil <= 0.2


class TestLookalikeClasses:
    """One test per look-alike class from the CLAUDE.md table."""

    def test_real_bilge_dump_accepted(self, model):
        v = model.classify(region(), WindContext.from_speed(5.5, 210))
        assert v.is_oil
        assert v.p_oil > 0.8

    def test_algal_bloom_rejected(self, model):
        v = model.classify(
            region(damping_db=3.5, elongation=1.6, compactness=0.82,
                   homogeneity=0.48, contrast=34.0),
            WindContext.from_speed(6.2, 180),
        )
        assert not v.is_oil

    def test_rain_cell_rejected(self, model):
        v = model.classify(
            region(damping_db=4.5, elongation=1.2, compactness=0.90,
                   homogeneity=0.55, contrast=45.0),
            WindContext.from_speed(7.0, 160),
        )
        assert not v.is_oil

    def test_weak_damping_rejected(self, model):
        """A patch barely darker than the sea is not a slick."""
        v = model.classify(region(damping_db=0.6), WindContext.from_speed(5.5))
        assert not v.is_oil
        assert v.gate_hit == "insufficient_damping"


class TestExplainability:
    """Every rejection must be sayable out loud."""

    def test_reason_is_human_readable(self, model):
        v = model.classify(
            region(elongation=1.3, compactness=0.88),
            WindContext.from_speed(5.5),
        )
        assert not v.is_oil
        assert len(v.rejected_reason) > 25
        assert not any(t in v.rejected_reason for t in ("tensor", "logit", "nan"))

    def test_contributions_are_exposed(self, model):
        v = model.classify(region(), WindContext.from_speed(5.5))
        assert "wind" in v.contributions
        assert "damping" in v.contributions
        assert "_bias" in v.contributions

    def test_marginal_wind_lowers_confidence(self, model):
        """Being at the window edge is evidence against a confident call."""
        mid = model.classify(region(), WindContext.from_speed(5.5))
        edge = model.classify(region(), WindContext.from_speed(2.6))
        assert edge.p_oil < mid.p_oil


class TestSinglePol:
    def test_missing_vh_degrades_gracefully(self, model):
        """VH is not always present; the stage must never crash without it."""
        v = model.classify(region(vh_vv=None), WindContext.from_speed(5.5))
        assert v.p_oil > 0.0
        assert v.contributions["vh_vv"] == 0.0


class TestPersistence:
    def test_round_trip(self, tmp_path, model):
        path = model.save(tmp_path / "lookalike.json")
        loaded = LookalikeModel.load(path)
        assert loaded.weights == model.weights
        assert loaded.bias == model.bias

    def test_missing_model_falls_back_loudly(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            m = LookalikeModel.load_or_prior(tmp_path / "absent.json")
        assert m.source == "prior"
        assert "prior" in caplog.text.lower()
