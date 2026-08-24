"""Tile screening, the classical detector, and internal-wave rejection."""

from __future__ import annotations

import numpy as np
import pytest

from detect.stage_a import screen_tiles, tile_darkness_score
from detect.stage_b import build_model, classical_dark_patch
from detect.wavetrain import find_wave_trains, wave_train_rejections
from detect.polygonize import RegionFeatures
from ingest.tiling import build_tiles


def sea_scene(shape=(2048, 2048), seed=0):
    return np.random.default_rng(seed).normal(-17.0, 0.8, shape).astype(np.float32)


def grid_for(vv):
    return build_tiles({"vv": vv}, np.zeros(vv.shape, bool),
                       tile_size=512, overlap=0.25, downsample=1,
                       bbox=(72.0, 15.0, 72.5, 15.5))


class TestStageAScreen:
    def test_empty_ocean_is_discarded(self):
        """Most of a scene is featureless water; the point is to skip it."""
        result = screen_tiles(grid_for(sea_scene()))
        assert result.stats["kept"] == 0

    def test_a_real_streak_is_kept(self):
        vv = sea_scene()
        vv[1000:1040, 400:1600] = -25.0
        result = screen_tiles(grid_for(vv))
        assert result.stats["kept"] > 0
        assert result.stats["reduction"] > 0.5

    def test_speckle_alone_does_not_trigger_it(self):
        """Raw speckle puts ~2% of pixels far below the median on any tile.

        A per-pixel test would therefore fire everywhere and screen nothing,
        which is why the score block-averages first. Stage A runs on the
        speckle-filtered copy, so ~0.8 dB is the spread it actually sees.
        """
        for seed in range(4):
            filtered_sea = sea_scene(seed=seed)
            assert screen_tiles(grid_for(filtered_sea)).stats["kept"] == 0

    def test_unfiltered_noise_fails_open_not_closed(self):
        """Deliberately over-inclusive: passing a boring tile costs one forward
        pass, dropping a real slick costs a missed spill. On abnormally noisy
        input the screen must keep tiles rather than silently discard them."""
        noisy = np.random.default_rng(3).normal(-17.0, 2.5, (2048, 2048)).astype(np.float32)
        result = screen_tiles(grid_for(noisy))
        assert result.stats["kept"] == result.stats["total"]

    def test_all_land_tiles_are_skipped(self):
        vv = sea_scene((1024, 1024))
        grid = build_tiles({"vv": vv}, np.ones((1024, 1024), bool),
                           tile_size=512, overlap=0.25, downsample=1)
        assert screen_tiles(grid).stats["kept"] == 0


class TestClassicalDetector:
    def test_finds_a_dark_streak(self):
        vv = np.random.default_rng(0).normal(-17, 0.8, (400, 400)).astype(np.float32)
        vv[190:200, 60:340] = -25.0
        prob = classical_dark_patch(vv, np.zeros((400, 400), bool))
        assert prob[190:200, 60:340].mean() > 0.8

    def test_open_sea_scores_low(self):
        vv = np.random.default_rng(0).normal(-17, 0.8, (400, 400)).astype(np.float32)
        prob = classical_dark_patch(vv, np.zeros((400, 400), bool))
        assert prob[50:100, 50:100].mean() < 0.4

    def test_land_is_never_flagged(self):
        vv = np.random.default_rng(0).normal(-17, 0.8, (400, 400)).astype(np.float32)
        land = np.zeros((400, 400), bool)
        land[:, 300:] = True
        assert classical_dark_patch(vv, land)[:, 300:].max() == 0.0

    def test_all_land_returns_empty(self):
        vv = np.full((100, 100), -17.0, np.float32)
        assert classical_dark_patch(vv, np.ones((100, 100), bool)).max() == 0.0


class TestModelConstruction:
    def test_default_is_unet_resnet34(self):
        """Cerulean's choice; ~24M params fine-tunes on a consumer GPU."""
        model = build_model("unet", "resnet34", in_channels=2, classes=5,
                            pretrained=False)
        params = sum(p.numel() for p in model.parameters())
        assert 20e6 < params < 30e6

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("not-a-real-net", pretrained=False)


def band(label, lat, elongation=8.0, area=10.0, orientation=90.0):
    return RegionFeatures(
        label=label, pixel_count=800, area_km2=area, centroid_rc=(0.0, 0.0),
        centroid_lonlat=(72.0, lat), elongation=elongation, compactness=0.12,
        orientation_deg=orientation, major_axis_km=10.0, minor_axis_km=1.2,
        damping_ratio=0.4, mean_db=-21.0, surround_db=-17.0,
        texture_homogeneity=0.5, texture_contrast=30.0, texture_variance=1.0,
        vh_vv_ratio=0.2, mean_confidence=0.7,
    )


class TestWaveTrains:
    def test_evenly_spaced_parallel_bands_are_a_train(self):
        """The signature of internal waves - and each band alone looks like oil."""
        bands = [band(i, 15.0 + i * 0.05) for i in range(5)]
        trains = find_wave_trains(bands)
        assert len(trains) == 1
        assert trains[0].size == 5
        assert trains[0].spacing_cv < 0.2

    def test_a_lone_streak_is_not_a_train(self):
        assert find_wave_trains([band(1, 15.0)]) == []

    def test_two_parallel_streaks_are_not_enough(self):
        """A vessel that turned can genuinely leave two parallel streaks."""
        assert find_wave_trains([band(1, 15.0), band(2, 15.05)]) == []

    def test_irregular_spacing_is_not_a_train(self):
        bands = [band(1, 15.0), band(2, 15.05), band(3, 15.5), band(4, 16.4)]
        assert find_wave_trains(bands) == []

    def test_non_parallel_features_are_not_a_train(self):
        bands = [band(i, 15.0 + i * 0.05, orientation=90.0 * (i % 2)) for i in range(5)]
        assert find_wave_trains(bands) == []

    def test_rejection_reason_names_the_evidence(self):
        bands = [band(i, 15.0 + i * 0.05) for i in range(5)]
        rejections = wave_train_rejections(bands)
        assert len(rejections) == 5
        reason = next(iter(rejections.values()))
        assert "internal-wave train" in reason
        assert "evenly spaced" in reason
