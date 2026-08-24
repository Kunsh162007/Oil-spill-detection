"""Train the segmentation network or the look-alike physics model.

    python scripts/train.py --config configs/train_baseline.yaml
    python scripts/train.py --config configs/train_baseline.yaml --stage lookalike

Non-negotiables, per CLAUDE.md:
  * the encoder always starts from pretrained weights, never random init
  * loss is Dice + focal, equally weighted - oil is well under 1% of pixels
    and plain cross-entropy converges to "sea everywhere" while reporting 99%
    accuracy
  * mixed precision throughout, gradient accumulation instead of big batches
  * writes checkpoints and results.json in a fixed schema
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config, resolve_path  # noqa: E402


class DiceFocalLoss:
    """Dice + focal, equally weighted.

    Dice handles the class imbalance directly by optimising overlap rather
    than per-pixel accuracy. Focal down-weights the overwhelming number of
    easy sea pixels so the gradient is not dominated by them.
    """

    def __init__(self, n_classes: int, gamma: float = 2.0, smooth: float = 1.0):
        self.n_classes = n_classes
        self.gamma = gamma
        self.smooth = smooth

    def __call__(self, logits, target):
        import torch
        import torch.nn.functional as F

        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dims)
        cardinality = probs.sum(dims) + onehot.sum(dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (cardinality + self.smooth)).mean()

        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target.long(), reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma * ce).mean()

        return dice + focal


class ManifestDataset:
    """Patches read on demand from full scenes via a JSON manifest.

    Materialising 21,744 patches as arrays costs ~5.7 GB; reading each window
    from its parent GeoTIFF costs nothing extra and is fast enough because
    rasterio pulls only the requested block off disk.

    Scenes are already Sigma0 dB, so the only preprocessing here is the same
    fixed-range normalisation the inference path uses - if these differ, the
    model sees a different distribution at test time than it trained on.
    """

    def __init__(self, manifest_path, augment: bool = False,
                 limit: int | None = None, in_channels: int = 2,
                 balance: bool = True):
        import json
        import random

        self.path = Path(manifest_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"No manifest at {self.path}. Run scripts/prepare_dataset.py first."
            )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        entries = data["entries"]

        # The set is ~91% oil-containing patches. Left alone the model learns
        # "there is always oil"; capping the majority restores a usable ratio.
        if balance:
            positives = [e for e in entries if e["class"] == 1]
            negatives = [e for e in entries if e["class"] == 0]
            if negatives:
                rng = random.Random(0)
                rng.shuffle(positives)
                keep = min(len(positives), max(len(negatives) * 3, 1))
                entries = positives[:keep] + negatives
                rng.shuffle(entries)

        self.entries = entries[:limit] if limit else entries
        self.patch = data.get("patch_size", 256)
        self.augment = augment
        self.in_channels = in_channels
        self._handles: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _open(self, relative_path: str):
        """Cache open rasterio handles - reopening per patch dominates runtime."""
        import rasterio

        if relative_path not in self._handles:
            self._handles[relative_path] = rasterio.open(REPO_ROOT / relative_path)
        return self._handles[relative_path]

    def __getitem__(self, idx: int):
        import numpy as np
        import rasterio
        import torch

        from ingest.calibrate import normalise_for_model

        entry = self.entries[idx]
        size = entry.get("size", self.patch)
        window = rasterio.windows.Window(entry["col"], entry["row"], size, size)

        image = self._open(entry["image"]).read(1, window=window, boundless=True, fill_value=-35.0)
        mask = self._open(entry["mask"]).read(1, window=window, boundless=True, fill_value=0.0)

        image = normalise_for_model(image.astype(np.float32))
        mask = (mask > 0.5).astype(np.int64)

        if image.shape != (size, size):
            pad = ((0, size - image.shape[0]), (0, size - image.shape[1]))
            image = np.pad(image, pad, constant_values=0.0)
            mask = np.pad(mask, pad, constant_values=0)

        image = np.repeat(image[None, ...], self.in_channels, axis=0)

        if self.augment:
            # Flips and 90-degree rotations only. Elastic or perspective
            # warping destroys the speckle statistics the model relies on.
            if np.random.rand() < 0.5:
                image, mask = image[:, :, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                image, mask = image[:, ::-1, :].copy(), mask[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                image = np.rot90(image, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k).copy()
            if np.random.rand() < 0.3:
                image = np.clip(image * np.random.uniform(0.92, 1.08), 0.0, 1.0)

        return torch.from_numpy(image.astype("float32")), torch.from_numpy(mask)


class PatchDataset:
    """Image/mask patches from a directory of .npy or image pairs.

    Layout expected (the Zenodo and SOS datasets both reduce to this):
        root/images/*.npy   float32 (C, H, W) Sigma0 dB, or (H, W)
        root/masks/*.npy    uint8   (H, W) class indices
    """

    def __init__(self, root: Path, tile_size: int = 512, augment: bool = False,
                 limit: int | None = None, in_channels: int = 2):
        import numpy as np

        self.root = Path(root)
        images = sorted((self.root / "images").glob("*.npy"))
        masks = sorted((self.root / "masks").glob("*.npy"))
        if not images:
            raise FileNotFoundError(
                f"No .npy patches under {self.root/'images'}. "
                f"Run scripts/prepare_dataset.py first."
            )
        if len(images) != len(masks):
            raise ValueError(f"{len(images)} images but {len(masks)} masks in {self.root}")

        self.pairs = list(zip(images, masks))[:limit] if limit else list(zip(images, masks))
        self.tile_size = tile_size
        self.augment = augment
        self.in_channels = in_channels
        self.np = np

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import numpy as np

        from ingest.calibrate import normalise_for_model

        img_path, mask_path = self.pairs[idx]
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)

        if img.ndim == 2:
            img = img[None, ...]
        img = normalise_for_model(img)

        if img.shape[0] < self.in_channels:
            img = np.repeat(img[:1], self.in_channels, axis=0)
        elif img.shape[0] > self.in_channels:
            img = img[: self.in_channels]

        if self.augment:
            # Flips and 90-degree rotations only. Elastic or perspective
            # warping destroys the speckle statistics the model relies on.
            if np.random.rand() < 0.5:
                img, mask = img[:, :, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k).copy()
            if np.random.rand() < 0.3:
                img = np.clip(img * np.random.uniform(0.9, 1.1), 0.0, 1.0)

        import torch

        return torch.from_numpy(img), torch.from_numpy(mask)


def iou_per_class(confusion):
    """Per-class IoU from a confusion matrix."""
    import numpy as np

    inter = np.diag(confusion).astype(np.float64)
    union = confusion.sum(1) + confusion.sum(0) - np.diag(confusion)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def train_segmentation(config, args) -> int:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from detect.stage_b import CLASS_NAMES, build_model

    tcfg = config.section("train")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    arch = tcfg.get("arch", config.get("detect.arch", "unet"))
    encoder = tcfg.get("encoder", config.get("detect.encoder", "resnet34"))
    n_classes = int(tcfg.get("classes", 5))
    in_channels = int(tcfg.get("in_channels", 2))
    epochs = int(args.epochs or tcfg.get("epochs", 30))
    batch_size = int(tcfg.get("batch_size", 4))
    accum = int(tcfg.get("grad_accum", 4))
    lr = float(tcfg.get("lr", 3e-4))

    run_name = args.run or f"{arch}-{encoder}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    run_dir = resolve_path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = tcfg.get("train_manifest")
    if train_manifest:
        train_ds = ManifestDataset(
            resolve_path(train_manifest), augment=True,
            limit=args.limit, in_channels=in_channels,
        )
        val_ds = ManifestDataset(
            resolve_path(tcfg.get("val_manifest", "data/dev/val_manifest.json")),
            augment=False, in_channels=in_channels,
            limit=int(tcfg.get("val_limit", 1200)),
        )
    else:
        train_ds = PatchDataset(resolve_path(tcfg.get("train_dir", "data/dev/train")),
                                augment=True, limit=args.limit, in_channels=in_channels)
        val_ds = PatchDataset(resolve_path(tcfg.get("val_dir", "data/dev/val")),
                              augment=False, in_channels=in_channels)

    print(f"Run           : {run_name}")
    print(f"Device        : {device}")
    print(f"Architecture  : {arch} / {encoder} (encoder pretrained)")
    print(f"Train / val   : {len(train_ds)} / {len(val_ds)} patches")
    print(f"Effective bs  : {batch_size} x {accum} = {batch_size*accum}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Never random init: a SAR encoder trained from scratch on ~2,850 patches
    # does not work, and a silently-random init is hard to spot in a loss curve.
    model = build_model(arch, encoder, in_channels, n_classes, pretrained=True).to(device)

    if tcfg.get("head_only", False):
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Head-only training: encoder frozen (fits a smaller GPU)")

    criterion = DiceFocalLoss(n_classes)
    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=device.startswith("cuda"))

    history, best_oil_iou = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        running, started = 0.0, time.perf_counter()
        optimiser.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.split(":")[0],
                                enabled=device.startswith("cuda")):
                loss = criterion(model(x), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
            running += float(loss.detach()) * accum
            if step % 25 == 0:
                print(f"    epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {float(loss.detach())*accum:.4f}", flush=True)

        model.eval()
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += float(criterion(logits, y))
                pred = logits.argmax(1).cpu().numpy().ravel()
                true = y.cpu().numpy().ravel()
                np.add.at(confusion, (true, pred), 1)

        ious = iou_per_class(confusion)
        oil_iou = float(ious[1]) if len(ious) > 1 and not np.isnan(ious[1]) else 0.0
        names = CLASS_NAMES[:n_classes] if n_classes <= len(CLASS_NAMES) else [
            f"class_{i}" for i in range(n_classes)
        ]
        row = {
            "epoch": epoch,
            "train_loss": round(running / max(len(train_dl), 1), 5),
            "val_loss": round(val_loss / max(len(val_dl), 1), 5),
            "oil_iou": round(oil_iou, 5),
            "mean_iou": round(float(np.nanmean(ious)), 5),
            "per_class_iou": {
                name: (None if np.isnan(v) else round(float(v), 5))
                for name, v in zip(names, ious)
            },
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(row)
        print(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  "
              f"val {row['val_loss']:.4f}  OIL IoU {row['oil_iou']:.4f}  "
              f"mIoU {row['mean_iou']:.4f}  ({row['seconds']}s)", flush=True)

        # Selected on OIL IoU, never mean IoU: mIoU is dominated by sea and
        # land, which every model scores in the 90s on.
        if oil_iou > best_oil_iou:
            best_oil_iou = oil_iou
            torch.save({
                "model_state": model.state_dict(),
                "config": {"arch": arch, "encoder": encoder,
                           "in_channels": in_channels, "classes": n_classes},
                "epoch": epoch, "oil_iou": oil_iou,
            }, run_dir / "best.pt")
        scheduler.step()

    results = {
        "run_name": run_name,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "arch": arch, "encoder": encoder, "device": device,
        "epochs": epochs, "n_train": len(train_ds), "n_val": len(val_ds),
        "best_oil_iou": round(best_oil_iou, 5),
        "selection_metric": "oil_iou (NOT mean IoU - mIoU is dominated by sea/land)",
        "history": history,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    final = resolve_path(config.get("detect.checkpoint", "models/stage_b.pt"))
    final.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy(run_dir / "best.pt", final)
    print(f"\nBest OIL IoU {best_oil_iou:.4f}")
    print(f"Results  -> {run_dir/'results.json'}")
    print(f"Deployed -> {final}")
    return 0


def train_lookalike(config, args) -> int:
    """Fit the interpretable look-alike model on labelled physical features.

    Expects a CSV whose columns are the feature names from
    detect.lookalike.transform_features plus a `label` column (1 = oil).
    Stays a logistic model on purpose: every coefficient has to remain
    explainable out loud.
    """
    import numpy as np

    from detect.lookalike import PRIOR_WEIGHTS, GateConfig, LookalikeModel, transform_features

    tcfg = config.section("train")
    features_csv = resolve_path(tcfg.get("lookalike_features", "data/dev/lookalike_features.csv"))
    if not features_csv.exists():
        print(
            f"No feature table at {features_csv}.\n"
            f"Build one with scripts/extract_features.py over the Yang et al.\n"
            f"look-alike dataset, then rerun. Until then the pipeline uses the\n"
            f"hand-set physical priors, which are stated as such at runtime.",
            file=sys.stderr,
        )
        return 1

    import csv as _csv

    rows = list(_csv.DictReader(features_csv.open(encoding="utf-8")))
    if not rows:
        print(f"{features_csv} is empty", file=sys.stderr)
        return 1

    names = list(PRIOR_WEIGHTS)
    X, y, clusters = [], [], []
    for row in rows:
        raw = {k: float(v) for k, v in row.items()
               if k not in ("label", "cluster") and v not in ("", None)}
        terms = transform_features(raw)
        X.append([terms.get(n, 0.0) for n in names])
        y.append(int(float(row["label"])))
        clusters.append(row.get("cluster", "unknown"))

    X, y = np.asarray(X), np.asarray(y)
    print(f"Fitting on {len(y)} samples ({y.sum()} oil, {len(y)-y.sum()} look-alike)")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te, _, c_te = train_test_split(
        X, y, clusters, test_size=0.3, random_state=int(args.seed), stratify=y
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    weights = dict(zip(names, clf.coef_[0].round(4).tolist()))
    bias = float(clf.intercept_[0])

    # Physics fixes the SIGN of each term. A fitted coefficient that flips one
    # means the feature table is mislabelled, and shipping it would produce a
    # model that rejects oil for looking like oil.
    expected_sign = {n: (1 if w > 0 else -1) for n, w in PRIOR_WEIGHTS.items()}
    flipped = [n for n, w in weights.items()
               if w != 0 and (1 if w > 0 else -1) != expected_sign[n]]
    if flipped:
        print(f"\nWARNING: these coefficients contradict the physics: {flipped}")
        print("Check the feature table labels before trusting this model.")

    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = float((preds == y_te).mean())

    # The decider is the FP rate per look-alike cluster, not overall accuracy.
    per_cluster: dict[str, dict] = {}
    for cluster in sorted(set(c_te)):
        mask = np.array([c == cluster for c in c_te])
        neg = mask & (y_te == 0)
        if neg.sum():
            per_cluster[cluster] = {
                "n": int(neg.sum()),
                "false_positive_rate": round(float(preds[neg].mean()), 4),
            }

    model = LookalikeModel(weights=weights, bias=bias,
                           gates=GateConfig(), source="fitted")
    out = resolve_path(config.get("lookalike.model_path", "models/lookalike.json"))
    model.save(out)

    print(f"\nHold-out accuracy: {accuracy:.4f}")
    print("False-positive rate per look-alike cluster (the primary decider):")
    for cluster, stats in sorted(per_cluster.items(), key=lambda kv: -kv[1]["false_positive_rate"]):
        print(f"  {cluster:28s} n={stats['n']:5d}  FP {stats['false_positive_rate']:.4f}")
    print(f"\nCoefficients:")
    for name, weight in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:22s} {weight:+.4f}")
    print(f"\nSaved -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["segmentation", "lookalike"], default="segmentation")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap training patches (few-shot)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--run", default=None, help="run name")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass

    config = load_config(args.config)
    if args.stage == "lookalike":
        return train_lookalike(config, args)
    return train_segmentation(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
