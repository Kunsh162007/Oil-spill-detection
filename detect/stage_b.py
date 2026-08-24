"""Stage B - segmentation.

Default architecture is a U-Net with an ImageNet-pretrained ResNet-34
encoder, via segmentation_models_pytorch. That is the same choice SkyTruth
Cerulean made for this exact task, which makes it defensible on stage, and
it behaves well on the ~2,850 pixel-labelled patches we actually have.

Classes (5):
    0 sea, 1 oil, 2 look-alike, 3 ship, 4 land

Until a checkpoint exists, the stage falls back to a classical adaptive
dark-patch detector. That fallback is NOT a fake: it genuinely finds dark
regions, so the physics stage downstream has real candidates to reason
about. It is simply weaker than a trained network, and says so.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ingest.calibrate import normalise_for_model
from ingest.pipeline import IngestResult
from ingest.tiling import TileGrid, merge_predictions

log = logging.getLogger(__name__)

CLASS_NAMES = ["sea", "oil", "lookalike", "ship", "land"]
OIL_CLASS = 1
LOOKALIKE_CLASS = 2
SHIP_CLASS = 3


@dataclass
class SegmentationResult:
    """Scene-sized probability maps plus how they were produced."""

    oil_probability: np.ndarray          # (H, W) float32
    lookalike_probability: np.ndarray | None
    ship_probability: np.ndarray | None
    backend: str                         # "unet-resnet34" | "classical-dark-patch"
    stats: dict[str, Any] = field(default_factory=dict)
    class_map: np.ndarray | None = None  # (H, W) uint8 argmax, when available


def build_model(arch: str = "unet", encoder: str = "resnet34", in_channels: int = 2,
                classes: int = 5, pretrained: bool = True):
    """Construct the segmentation network.

    The encoder is ALWAYS pretrained unless explicitly disabled - training a
    SAR segmentation encoder from scratch on 2,850 patches does not work, and
    a silently-random init is hard to spot in a loss curve.
    """
    import segmentation_models_pytorch as smp

    weights = "imagenet" if pretrained else None
    factory = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "segformer": getattr(smp, "Segformer", None),
    }.get(arch.lower())
    if factory is None:
        raise ValueError(f"Unknown architecture {arch!r}; known: unet, unetplusplus, deeplabv3plus, fpn, segformer")

    return factory(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=in_channels,
        classes=classes,
    )


def _tile_to_input(tile_data: np.ndarray, in_channels: int) -> np.ndarray:
    """Normalise a tile to [0,1] and force it to the expected channel count.

    Single-pol scenes duplicate VV rather than zero-filling VH: a zero channel
    shifts the input distribution away from anything the model saw in
    training, and the network reads that as signal.
    """
    x = normalise_for_model(tile_data)
    if x.shape[0] == in_channels:
        return x
    if x.shape[0] == 1 and in_channels > 1:
        return np.repeat(x, in_channels, axis=0)
    if x.shape[0] > in_channels:
        return x[:in_channels]
    pad = np.repeat(x[:1], in_channels - x.shape[0], axis=0)
    return np.concatenate([x, pad], axis=0)


class SegmentationModel:
    """Loaded network plus batched, mixed-precision tile inference."""

    def __init__(self, model, device: str = "cpu", in_channels: int = 2,
                 classes: int = 5, use_fp16: bool = True) -> None:
        import torch

        self.torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.in_channels = in_channels
        self.classes = classes
        self.use_fp16 = use_fp16 and device.startswith("cuda")

    @classmethod
    def load(cls, checkpoint: str | Path, arch: str = "unet", encoder: str = "resnet34",
             in_channels: int = 2, classes: int = 5, device: str | None = None) -> SegmentationModel:
        import torch

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = state.get("config", {}) if isinstance(state, dict) else {}
        arch = cfg.get("arch", arch)
        encoder = cfg.get("encoder", encoder)
        in_channels = cfg.get("in_channels", in_channels)
        classes = cfg.get("classes", classes)

        model = build_model(arch, encoder, in_channels, classes, pretrained=False)
        weights = state.get("model_state", state) if isinstance(state, dict) else state
        model.load_state_dict(weights)
        log.info("Loaded %s/%s from %s onto %s", arch, encoder, ckpt_path.name, device)
        return cls(model, device=device, in_channels=in_channels, classes=classes)

    def predict_tiles(self, grid: TileGrid, batch_size: int = 8,
                      keep_indices: list[int] | None = None) -> list[np.ndarray]:
        """Run inference over tiles, returning one (C, H, W) probability map each.

        Tiles not in keep_indices (screened out by stage A) are filled with
        all-sea probability rather than skipped, so merge_predictions still
        sees a complete grid.
        """
        torch = self.torch
        keep = set(range(len(grid.tiles))) if keep_indices is None else set(keep_indices)

        empty = np.zeros((self.classes, grid.tile_size, grid.tile_size), dtype=np.float32)
        empty[0] = 1.0  # all sea
        outputs: list[np.ndarray] = [empty.copy() for _ in grid.tiles]

        todo = [t for t in grid.tiles if t.index in keep]
        if not todo:
            return outputs

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_fp16
            else _null_context()
        )

        with torch.no_grad():
            for start in range(0, len(todo), batch_size):
                chunk = todo[start : start + batch_size]
                batch = np.stack([_tile_to_input(t.data, self.in_channels) for t in chunk])
                x = torch.from_numpy(batch).to(self.device)
                with autocast:
                    logits = self.model(x)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                for tile, p in zip(chunk, probs):
                    # Land and nodata can never be oil; zero them before merging.
                    p = p.copy()
                    p[:, ~tile.valid] = 0.0
                    p[0, ~tile.valid] = 1.0
                    outputs[tile.index] = p.astype(np.float32)
        return outputs


class _null_context:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def classical_dark_patch(
    sigma0_db: np.ndarray,
    land_mask: np.ndarray,
    sensitivity: float = 2.0,
    multiscale: bool = True,
) -> np.ndarray:
    """Adaptive dark-patch detector - the fallback when no checkpoint exists.

    Flags pixels sitting well below the local background. This is close to the
    pre-deep-learning operational approach: a genuine detector, not a stub. It
    cannot tell oil from a look-alike, but separating those is the physics
    stage's job anyway.

    Three things make it markedly more accurate than a plain global threshold:

    1. MULTI-SCALE background. A slick has no single size. Comparing against
       several background windows and taking the strongest response finds a
       narrow bilge streak and a broad drifting patch with the same settings,
       where one window size only ever suits one of them.
    2. ROBUST statistics. Median and MAD rather than mean and standard
       deviation, because the slick itself is in the data being summarised
       and would otherwise drag the "background" down towards itself.
    3. GRADIENT confirmation. Real slicks have an edge. Weighting by local
       contrast suppresses the broad, soft dimming that low wind produces
       across a whole region - which is the single most common false alarm.
    """
    from scipy import ndimage

    arr = np.asarray(sigma0_db, dtype=np.float32)
    sea = ~land_mask
    if not sea.any():
        return np.zeros_like(arr, dtype=np.float32)

    valid = sea & (arr > -90.0)
    if not valid.any():
        return np.zeros_like(arr, dtype=np.float32)

    # Fill land and nodata with the sea median so they neither read as dark
    # structure nor drag the background down at the coastline.
    median = float(np.median(arr[valid]))
    filled = np.where(valid, arr, median).astype(np.float32)

    residual = np.abs(filled[valid] - np.median(filled[valid]))
    mad = float(np.median(residual))
    robust_std = max(1.4826 * mad, 0.25)

    # Windows spanning roughly 2 km to 16 km at 80 m/px.
    scales = (25, 61, 121, 201) if multiscale else (101,)
    best = np.zeros_like(filled, dtype=np.float32)
    for size in scales:
        if size >= min(filled.shape):
            continue
        background = ndimage.uniform_filter(filled, size=size, mode="reflect")
        z = (background - filled) / robust_std   # positive where darker
        best = np.maximum(best, z)

    prob = 1.0 / (1.0 + np.exp(-(best - sensitivity)))

    # Edge confirmation. A slick boundary is a real gradient; a low-wind
    # region dims gradually and produces almost none.
    gradient = ndimage.gaussian_gradient_magnitude(filled, sigma=2.0)
    edge_scale = max(float(np.percentile(gradient[valid], 90)), 1e-3)
    edge = np.clip(gradient / edge_scale, 0.0, 1.0)
    # Nearby edge, not edge at the exact pixel: the interior of a slick is
    # flat, and only its rim carries the gradient.
    edge_support = ndimage.maximum_filter(edge, size=15)

    prob = prob * (0.55 + 0.45 * edge_support)

    prob = np.where(valid, prob, 0.0)
    return prob.astype(np.float32)


def run_stage_b(
    ingest: IngestResult,
    config,
    keep_indices: list[int] | None = None,
) -> SegmentationResult:
    """Segment a scene, using the network when available and the classical
    detector when not."""
    det = config.section("detect")
    checkpoint = det.get("checkpoint", "models/stage_b.pt")
    from core.config import resolve_path

    ckpt_path = resolve_path(checkpoint)
    forced_stub = config.use_stub("stage_b")
    started = time.perf_counter()

    if not forced_stub and ckpt_path.exists():
        try:
            model = SegmentationModel.load(
                ckpt_path,
                arch=det.get("arch", "unet"),
                encoder=det.get("encoder", "resnet34"),
                in_channels=len(ingest.grid.meta.get("channels", ["vv"])),
                classes=int(det.get("classes", 5)),
            )
            preds = model.predict_tiles(
                ingest.grid,
                batch_size=int(det.get("batch_size", 8)),
                keep_indices=keep_indices,
            )
            merged = merge_predictions(ingest.grid, preds)  # (C, H, W)
            # A model trained on binary masks has two classes, not five. Only
            # index the heads it actually has: oil is class 1 under either
            # scheme, but look-alike and ship exist only in the 5-class one.
            n_classes = merged.shape[0]
            elapsed = time.perf_counter() - started
            return SegmentationResult(
                oil_probability=merged[OIL_CLASS],
                lookalike_probability=(
                    merged[LOOKALIKE_CLASS] if n_classes > LOOKALIKE_CLASS else None
                ),
                ship_probability=(
                    merged[SHIP_CLASS] if n_classes > SHIP_CLASS else None
                ),
                backend=f"{det.get('arch','unet')}-{det.get('encoder','resnet34')}",
                class_map=np.argmax(merged, axis=0).astype(np.uint8),
                stats={
                    "elapsed_s": round(elapsed, 2),
                    "device": model.device,
                    "classes": n_classes,
                    "tiles_run": len(keep_indices) if keep_indices else len(ingest.grid),
                    "checkpoint": str(ckpt_path),
                },
            )
        except Exception as exc:
            # A broken checkpoint must not take the demo down, but it must be
            # loud - a silent downgrade to the weaker detector would be
            # reported as a model result.
            log.error(
                "Segmentation network failed (%s). Falling back to the classical "
                "detector; results are WEAKER than a trained model.", exc,
            )
    elif not forced_stub:
        log.warning(
            "No checkpoint at %s - using the classical dark-patch detector. "
            "Run scripts/train.py to produce one.", ckpt_path,
        )

    prob = classical_dark_patch(
        ingest.sigma0_db["vv"], ingest.land_mask,
        sensitivity=float(det.get("classical_sensitivity", 2.0)),
    )
    elapsed = time.perf_counter() - started
    return SegmentationResult(
        oil_probability=prob,
        lookalike_probability=None,
        ship_probability=None,
        backend="classical-dark-patch",
        stats={
            "elapsed_s": round(elapsed, 2),
            "note": (
                "Classical adaptive threshold, not a trained network. Finds dark "
                "patches; relies entirely on the physics stage to reject look-alikes."
            ),
        },
    )
