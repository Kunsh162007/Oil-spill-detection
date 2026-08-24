"""Turn a downloaded oil-spill dataset into a training manifest.

Deliberately does NOT materialise patches to disk. The Zenodo Gulf of Mexico
set defines 21,744 training windows over 14 full scenes; writing those out as
arrays needs ~5.7 GB, while reading each window from its parent GeoTIFF on
demand needs none. rasterio windowed reads make that cheap.

Output is one JSON manifest per split listing (scene, row, col, class), which
scripts/train.py consumes directly.

    python scripts/prepare_dataset.py --source zenodo-gom
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def prepare_zenodo_gom(raw_dir: Path, out_dir: Path, patch: int = 256) -> int:
    """Zenodo record 4672426 - Gulf of Mexico oil spill segmentation.

    23 Sentinel-1 scenes (2018-2020) in Sigma0 dB with binary oil masks,
    cross-referenced against NOAA incident reports. CC-BY-4.0.
    """
    import pandas as pd
    import rasterio

    train_csv = raw_dir / "train" / "dataframe_train_dataset_256_90.csv"
    val_csv = raw_dir / "train" / "dataframe_val_dataset_256_90.csv"
    if not train_csv.exists():
        raise SystemExit(
            f"Expected {train_csv}. Download Zenodo record 4672426 and extract "
            f"it to {raw_dir}."
        )

    def build(csv_path: Path, split: str) -> dict:
        df = pd.read_csv(csv_path)
        entries, missing = [], Counter()

        for _, row in df.iterrows():
            # The CSV carries the original author's Windows paths; only the
            # filename is portable.
            scene_name = str(row["paths"]).replace("\\", "/").rsplit("/", 1)[-1]
            image = raw_dir / "train" / "images" / scene_name
            mask = raw_dir / "train" / "masks" / scene_name
            if not image.exists() or not mask.exists():
                missing[scene_name] += 1
                continue
            # Verified empirically against the masks: the first value is the
            # COLUMN and the pair marks the patch CENTRE, not its top-left.
            # Reading it as (row, col) top-left drops the share of oil-bearing
            # positive patches from 66% to 21% - i.e. most labels would be
            # wrong, and the model would learn from noise.
            cx, cy = (int(v) for v in str(row["coordinates"]).split(","))
            entries.append({
                "image": str(image.relative_to(REPO_ROOT)),
                "mask": str(mask.relative_to(REPO_ROOT)),
                "row": cy - patch // 2,
                "col": cx - patch // 2,
                "size": patch,
                "class": int(float(row["class"])),
            })

        if missing:
            print(f"  {split}: skipped {sum(missing.values())} rows for "
                  f"{len(missing)} missing scene(s)")
        return {
            "split": split,
            "patch_size": patch,
            "n": len(entries),
            "class_counts": dict(Counter(e["class"] for e in entries)),
            "entries": entries,
        }

    def verify(manifest: dict, sample: int = 200) -> float:
        """Fraction of positive patches that genuinely contain oil pixels.

        A cheap guard against silently mis-decoding the coordinate convention,
        which is the single easiest way to train on garbage here.
        """
        import random

        positives = [e for e in manifest["entries"] if e["class"] == 1]
        if not positives:
            return 0.0
        picked = random.Random(0).sample(positives, min(sample, len(positives)))
        hits = 0
        for entry in picked:
            with rasterio.open(REPO_ROOT / entry["mask"]) as src:
                window = rasterio.windows.Window(
                    entry["col"], entry["row"], entry["size"], entry["size"]
                )
                data = src.read(1, window=window, boundless=True, fill_value=0.0)
            hits += int((data > 0.5).any())
        return hits / len(picked)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for csv_path, split in ((train_csv, "train"), (val_csv, "val")):
        manifest = build(csv_path, split)
        quality = verify(manifest)
        manifest["positive_patches_with_oil"] = round(quality, 4)
        if quality < 0.5:
            print(f"  WARNING: only {quality:.0%} of {split} positive patches contain "
                  f"oil pixels - the coordinate convention may be misread")
        path = out_dir / f"{split}_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        total += manifest["n"]
        print(f"  {split}: {manifest['n']} patches, classes {manifest['class_counts']}, "
              f"{quality:.0%} of positives contain oil -> {path.name}")

    # The held-out test scenes are whole rasters, not windows: keep them as a
    # scene list so evaluation runs on complete images, the way the pipeline
    # actually sees them.
    test_images = sorted((raw_dir / "test" / "images").glob("*.tif"))
    test_manifest = {
        "split": "test",
        "scenes": [
            {
                "image": str(p.relative_to(REPO_ROOT)),
                "mask": str((raw_dir / "test" / "masks" / p.name).relative_to(REPO_ROOT)),
            }
            for p in test_images
            if (raw_dir / "test" / "masks" / p.name).exists()
        ],
    }
    (out_dir / "test_manifest.json").write_text(json.dumps(test_manifest), encoding="utf-8")
    print(f"  test: {len(test_manifest['scenes'])} held-out scenes -> test_manifest.json")

    with rasterio.open(test_images[0]) as src:
        print(f"\n  scene format: {src.count} band(s), {src.dtypes[0]}, CRS {src.crs}")
    print(f"\nTotal {total} training/validation patches, read on demand (0 GB extra).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="zenodo-gom", choices=["zenodo-gom"])
    ap.add_argument("--raw-dir", default="data/raw/extracted")
    ap.add_argument("--out-dir", default="data/dev")
    ap.add_argument("--patch", type=int, default=256)
    args = ap.parse_args()

    raw = REPO_ROOT / args.raw_dir
    out = REPO_ROOT / args.out_dir
    print(f"Preparing {args.source} from {raw}")
    return prepare_zenodo_gom(raw, out, args.patch)


if __name__ == "__main__":
    raise SystemExit(main())
