"""Downsampling and tiling.

Two decisions live here and they matter more than any model choice
(CLAUDE.md, "The two decisions that matter more than the model"):

  1. Work at ~80 m/px, not 10 m. Slicks are hundreds of metres to kilometres
     across, so 10 m detail is speckle, not signal. An 8x block-mean is 64x
     fewer pixels AND suppresses speckle by sqrt(64) = 8.
  2. Tile 512 with 25% overlap so no slick is cut by a tile seam, then merge
     predictions by averaging confidence in the overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ingest.calibrate import db_to_linear

S1_IW_GRD_PIXEL_M = 10.0


@dataclass
class Tile:
    """One 512x512 window, with its position in the parent (downsampled) array."""

    row: int          # top edge, in downsampled pixels
    col: int          # left edge
    data: np.ndarray  # (C, H, W) float32
    valid: np.ndarray  # (H, W) bool - False on land/nodata
    index: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[-2:]

    @property
    def sea_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


@dataclass
class TileGrid:
    """The full set of tiles for a scene, plus what is needed to stitch back."""

    tiles: list[Tile]
    full_shape: tuple[int, int]     # downsampled (H, W)
    tile_size: int
    stride: int
    downsample_factor: int
    resolution_m: float
    bbox: tuple[float, float, float, float] | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tiles)


def downsample_factor_for(
    source_res_m: float = S1_IW_GRD_PIXEL_M, target_res_m: float = 80.0
) -> int:
    """Integer block factor to reach the target resolution (min 1)."""
    return max(1, int(round(target_res_m / source_res_m)))


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping factor x factor blocks, trimming any remainder.

    Averaging happens in LINEAR power, not dB. Averaging decibels is averaging
    logarithms, which biases every result low and is a classic SAR mistake.
    """
    if factor <= 1:
        return np.asarray(arr, dtype=np.float32)
    a = np.asarray(arr, dtype=np.float64)
    h, w = a.shape[-2:]
    h_trim, w_trim = (h // factor) * factor, (w // factor) * factor
    a = a[..., :h_trim, :w_trim]
    new_shape = a.shape[:-2] + (h_trim // factor, factor, w_trim // factor, factor)
    return a.reshape(new_shape).mean(axis=(-3, -1)).astype(np.float32)


def downsample_db(db: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a dB array correctly: to linear, average, back to dB."""
    if factor <= 1:
        return np.asarray(db, dtype=np.float32)
    linear = db_to_linear(db)
    averaged = block_mean(linear, factor)
    return (10.0 * np.log10(np.maximum(averaged, 1e-12))).astype(np.float32)


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a boolean mask. A block is masked if ANY source pixel was.

    Deliberately conservative: it is far better to discard a few genuine sea
    pixels near a coast than to admit a land pixel into a slick polygon.
    """
    if factor <= 1:
        return np.asarray(mask, dtype=bool)
    m = block_mean(np.asarray(mask, dtype=np.float32), factor)
    return m > 0.0


def iter_windows(
    shape: tuple[int, int], tile_size: int, overlap: float
) -> Iterator[tuple[int, int]]:
    """Yield (row, col) top-left corners covering the shape with overlap.

    The last row/column is snapped back to the array edge so the far margin is
    always covered rather than silently dropped.
    """
    height, width = shape
    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    rows = list(range(0, max(1, height - tile_size + 1), stride))
    if not rows or rows[-1] + tile_size < height:
        rows.append(max(0, height - tile_size))
    cols = list(range(0, max(1, width - tile_size + 1), stride))
    if not cols or cols[-1] + tile_size < width:
        cols.append(max(0, width - tile_size))

    for r in dict.fromkeys(rows):
        for c in dict.fromkeys(cols):
            yield r, c


def _pad_to(arr: np.ndarray, size: int, fill: float) -> np.ndarray:
    """Pad the trailing two dims out to size (edge tiles on a small scene)."""
    h, w = arr.shape[-2:]
    if h >= size and w >= size:
        return arr
    pad = [(0, 0)] * (arr.ndim - 2) + [(0, max(0, size - h)), (0, max(0, size - w))]
    return np.pad(arr, pad, mode="constant", constant_values=fill)


def build_tiles(
    channels: dict[str, np.ndarray],
    land_mask: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.25,
    downsample: int = 8,
    source_res_m: float = S1_IW_GRD_PIXEL_M,
    bbox: tuple[float, float, float, float] | None = None,
) -> TileGrid:
    """Downsample the scene and cut it into overlapping tiles.

    The channels mapping goes from polarisation name to a dB array, keyed
    vv and vh. Single-pol scenes are handled by the caller duplicating VV;
    see ingest.pipeline.
    """
    if not channels:
        raise ValueError("build_tiles requires at least one channel")

    names = list(channels)
    small = {k: downsample_db(v, downsample) for k, v in channels.items()}
    small_mask = downsample_mask(land_mask, downsample)

    shapes = {k: v.shape for k, v in small.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Channel shapes disagree after downsampling: {shapes}")
    height, width = small[names[0]].shape
    if small_mask.shape != (height, width):
        raise ValueError(
            f"Land mask shape {small_mask.shape} != image shape {(height, width)}"
        )

    stacked = np.stack([small[k] for k in names], axis=0)  # (C, H, W)
    sea = ~small_mask

    tiles: list[Tile] = []
    for idx, (r, c) in enumerate(iter_windows((height, width), tile_size, overlap)):
        data = stacked[:, r : r + tile_size, c : c + tile_size]
        valid = sea[r : r + tile_size, c : c + tile_size]
        data = _pad_to(data, tile_size, fill=-99.0)
        valid = _pad_to(valid.astype(np.float32), tile_size, fill=0.0).astype(bool)
        tiles.append(Tile(row=r, col=c, data=data, valid=valid, index=idx))

    return TileGrid(
        tiles=tiles,
        full_shape=(height, width),
        tile_size=tile_size,
        stride=max(1, int(round(tile_size * (1.0 - overlap)))),
        downsample_factor=downsample,
        resolution_m=source_res_m * downsample,
        bbox=bbox,
        meta={"channels": names, "n_tiles": len(tiles)},
    )


def merge_predictions(grid: TileGrid, predictions: list[np.ndarray]) -> np.ndarray:
    """Stitch per-tile probability maps back into one scene-sized map.

    Overlapping pixels are averaged (weight-normalised), which is what makes
    the 25% overlap worth paying for: a slick straddling a seam gets a smooth
    confidence field instead of a visible discontinuity.
    """
    if len(predictions) != len(grid.tiles):
        raise ValueError(
            f"Got {len(predictions)} predictions for {len(grid.tiles)} tiles"
        )

    height, width = grid.full_shape
    sample = predictions[0]
    n_class = sample.shape[0] if sample.ndim == 3 else 1

    accum = np.zeros((n_class, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)

    for tile, pred in zip(grid.tiles, predictions):
        p = pred if pred.ndim == 3 else pred[None, ...]
        h = min(grid.tile_size, height - tile.row)
        w = min(grid.tile_size, width - tile.col)
        accum[:, tile.row : tile.row + h, tile.col : tile.col + w] += p[:, :h, :w]
        weight[tile.row : tile.row + h, tile.col : tile.col + w] += 1.0

    weight = np.maximum(weight, 1e-6)
    merged = accum / weight[None, ...]
    return merged[0] if n_class == 1 else merged


def pixel_to_lonlat(
    row: float,
    col: float,
    shape: tuple[int, int],
    bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Map an array pixel to (lon, lat). Row 0 is the north edge."""
    height, width = shape
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + (col / max(1, width - 1)) * (max_lon - min_lon)
    lat = max_lat - (row / max(1, height - 1)) * (max_lat - min_lat)
    return (lon, lat)
