"""Ingest: calibration, speckle, land mask, tiling.

Uses the real synthetic GeoTIFF from data/demo_internal, not in-memory fakes,
so the rasterio path is genuinely exercised.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ingest.calibrate import db_to_linear, normalise_for_model, to_sigma0_db
from ingest.landmask import land_mask_for_bbox
from ingest.speckle import filter_pair, refined_lee, speckle_index
from ingest.tiling import (
    block_mean,
    build_tiles,
    downsample_db,
    merge_predictions,
    pixel_to_lonlat,
)


class TestCalibration:
    def test_detects_raw_dn(self):
        dn = np.random.default_rng(0).gamma(4, 60, (64, 64))
        assert to_sigma0_db(dn).mode == "raw_dn_generic"

    def test_detects_existing_sigma0(self):
        s0 = np.random.default_rng(0).gamma(4, 0.02, (64, 64))
        assert to_sigma0_db(s0).mode == "already_sigma0"

    def test_db_linear_round_trip(self):
        values = np.array([0.001, 0.02, 0.5], dtype=np.float64)
        db = 10 * np.log10(values)
        assert np.allclose(db_to_linear(db), values)

    def test_normalisation_uses_fixed_physical_bounds(self):
        """Per-tile autoscaling would make an all-oil tile look like open sea."""
        dark = np.full((8, 8), -30.0, dtype=np.float32)
        bright = np.full((8, 8), -10.0, dtype=np.float32)
        assert normalise_for_model(dark).mean() < normalise_for_model(bright).mean()


class TestSpeckle:
    def test_filtering_reduces_speckle(self):
        rng = np.random.default_rng(1)
        truth = np.full((256, 256), 0.08)
        noisy = truth * rng.gamma(4.4, 1 / 4.4, truth.shape)
        assert speckle_index(refined_lee(noisy)) < speckle_index(noisy) * 0.7

    def test_slick_edge_contrast_survives(self):
        """Refined Lee must smooth along edges, not across them."""
        rng = np.random.default_rng(1)
        truth = np.full((256, 256), 0.08)
        truth[100:140, 40:220] = 0.02
        noisy = truth * rng.gamma(4.4, 1 / 4.4, truth.shape)
        filtered = refined_lee(noisy)

        def contrast(a):
            return a[110:130, 60:200].mean() / a[10:30, 60:200].mean()

        assert contrast(filtered) == pytest.approx(contrast(truth), abs=0.05)

    def test_light_copy_is_less_filtered(self):
        """Texture features must not be measured on the aggressive copy."""
        rng = np.random.default_rng(2)
        img = np.full((128, 128), 0.05) * rng.gamma(4.4, 1 / 4.4, (128, 128))
        heavy, light = filter_pair(img)
        assert speckle_index(light) > speckle_index(heavy)


class TestTiling:
    def test_db_downsampling_happens_in_linear_domain(self):
        """Averaging decibels averages logarithms and biases every result low."""
        db = np.array([[-20.0, -10.0], [-20.0, -10.0]], dtype=np.float32)
        proper = float(downsample_db(db, 2)[0, 0])
        naive = float(db.mean())
        assert proper > naive

    def test_block_mean_trims_remainder(self):
        assert block_mean(np.ones((10, 10)), 4).shape == (2, 2)

    def test_tiles_cover_the_scene(self):
        vv = np.full((1200, 1200), -17.0, dtype=np.float32)
        grid = build_tiles({"vv": vv}, np.zeros((1200, 1200), bool),
                           tile_size=512, overlap=0.25, downsample=1,
                           bbox=(72.0, 15.0, 72.5, 15.5))
        covered = np.zeros(grid.full_shape, bool)
        for t in grid.tiles:
            covered[t.row:t.row + 512, t.col:t.col + 512] = True
        assert covered.all()

    def test_merge_preserves_uniform_confidence(self):
        vv = np.full((900, 900), -17.0, dtype=np.float32)
        grid = build_tiles({"vv": vv}, np.zeros((900, 900), bool),
                           tile_size=512, overlap=0.25, downsample=1)
        preds = [np.full((1, 512, 512), 0.7, np.float32) for _ in grid.tiles]
        merged = merge_predictions(grid, preds)
        assert np.allclose(merged, 0.7, atol=1e-5)

    def test_geolocation_corners(self):
        """Row 0 is the NORTH edge - getting this wrong lands everything in
        the wrong ocean."""
        bbox = (72.0, 15.0, 72.6, 15.6)
        assert pixel_to_lonlat(0, 0, (100, 100), bbox) == pytest.approx((72.0, 15.6))
        assert pixel_to_lonlat(99, 99, (100, 100), bbox) == pytest.approx((72.6, 15.0))

    def test_channel_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="disagree"):
            build_tiles(
                {"vv": np.zeros((100, 100), np.float32),
                 "vh": np.zeros((90, 90), np.float32)},
                np.zeros((100, 100), bool), downsample=1,
            )


class TestLandMask:
    def test_open_ocean_is_not_land(self):
        assert not land_mask_for_bbox((68.0, 12.0, 69.0, 13.0), (50, 50)).any()

    def test_coast_is_partly_land(self):
        mask = land_mask_for_bbox((75.5, 8.8, 76.8, 9.9), (100, 100))
        assert 0.1 < mask.mean() < 0.9

    def test_north_up_orientation(self):
        """West of the Kerala coast is sea; east is land."""
        mask = land_mask_for_bbox((75.5, 8.8, 76.8, 9.9), (100, 100))
        assert mask[:, -1].mean() > mask[:, 0].mean()


class TestRealScene:
    def test_ingest_of_the_demo_geotiff(self, demo_scene, demo_config):
        """The full ingest path on a real file on disk."""
        from ingest.pipeline import ingest_scene

        result = ingest_scene(demo_scene, demo_config, use_cache=False)
        assert result.grid.tiles
        assert "vv" in result.sigma0_db and "vh" in result.sigma0_db
        assert result.is_dual_pol
        # Native pixel is ~47 m, so the 80 m target must not decimate by 8.
        assert 40.0 < result.grid.resolution_m < 200.0
        assert result.sigma0_db["vv"].shape == result.land_mask.shape

    def test_cache_returns_an_equivalent_result(self, demo_scene, demo_config):
        from ingest.pipeline import ingest_scene

        first = ingest_scene(demo_scene, demo_config, use_cache=True)
        second = ingest_scene(demo_scene, demo_config, use_cache=True)
        assert first.grid.full_shape == second.grid.full_shape
        assert np.allclose(first.sigma0_db["vv"], second.sigma0_db["vv"])
