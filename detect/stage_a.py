"""Stage A - the cheap tile screen.

Most of an ocean scene contains nothing. This discards those tiles before the
segmentation network sees them, so stage B runs on roughly a tenth of the
scene. Together with the 80 m resolution decision that is worth more than any
backbone choice.

Built LAST on purpose (CLAUDE.md build order): it is a speed optimisation, and
a wrong screen silently deletes real slicks. It is therefore deliberately
over-inclusive - the cost of passing a boring tile through is one forward
pass; the cost of dropping a real one is a missed spill.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ingest.tiling import Tile, TileGrid

log = logging.getLogger(__name__)


@dataclass
class ScreenResult:
    keep_indices: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


def tile_darkness_score(tile: Tile) -> float:
    """Cheap statistic: does this tile contain a sustained dark region?

    Block-averages first. That matters: raw SAR speckle puts ~2% of pixels
    far below the median on any tile of empty sea, so a per-pixel test fires
    everywhere and screens nothing. Averaging 8x8 blocks divides the speckle
    spread by 8 while leaving a real slick - which is hundreds of metres
    across and therefore many whole blocks wide - essentially untouched.
    """
    vv = tile.data[0]
    sea = tile.valid & (vv > -90.0)
    if sea.sum() < 256:
        return 0.0

    # Block-average to kill speckle. Masked pixels are filled with the sea
    # median so land edges do not read as dark structure.
    median_all = float(np.median(vv[sea]))
    filled = np.where(sea, vv, median_all)
    blocked = _block_mean_2d(filled, 8)
    if blocked.size < 16:
        return 0.0

    values = blocked.ravel()
    median = float(np.median(values))
    p25, p75 = np.percentile(values, [25, 75])
    spread = max(float(p75 - p25) / 1.349, 0.15)  # robust sigma, floored

    z = (median - values) / spread          # positive = darker than typical
    dark_fraction = float(np.mean(z > 2.5))  # blocks clearly below the sea level
    max_depth = float(np.max(z)) if values.size else 0.0

    # Either a decent patch of moderately dark blocks, or a few very dark ones.
    area_term = min(dark_fraction / 0.02, 1.0)
    depth_term = min(max(max_depth - 2.5, 0.0) / 4.0, 1.0)
    return float(np.clip(0.7 * area_term + 0.3 * depth_term, 0.0, 1.0))


def _block_mean_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping blocks, trimming any remainder."""
    h, w = arr.shape
    h_t, w_t = (h // factor) * factor, (w // factor) * factor
    if h_t == 0 or w_t == 0:
        return arr
    a = arr[:h_t, :w_t].astype(np.float64)
    return a.reshape(h_t // factor, factor, w_t // factor, factor).mean(axis=(1, 3))


def screen_tiles(
    grid: TileGrid, threshold: float = 0.15, min_sea_fraction: float = 0.05
) -> ScreenResult:
    """Select which tiles are worth running the network on."""
    started = time.perf_counter()
    keep: list[int] = []
    scores: dict[int, float] = {}

    for tile in grid.tiles:
        if tile.sea_fraction < min_sea_fraction:
            scores[tile.index] = 0.0
            continue  # essentially all land
        score = tile_darkness_score(tile)
        scores[tile.index] = round(score, 4)
        if score >= threshold:
            keep.append(tile.index)

    elapsed = time.perf_counter() - started
    kept = len(keep)
    total = len(grid.tiles)
    log.info(
        "Stage A kept %d/%d tiles (%.0f%%) in %.3fs",
        kept, total, 100.0 * kept / max(total, 1), elapsed,
    )
    return ScreenResult(
        keep_indices=keep,
        scores=scores,
        stats={
            "kept": kept,
            "total": total,
            "reduction": round(1.0 - kept / max(total, 1), 3),
            "elapsed_s": round(elapsed, 4),
            "threshold": threshold,
        },
    )
