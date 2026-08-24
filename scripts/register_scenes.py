"""Register real SAR scenes so the pipeline can analyse them.

Writes one manifest per scene into a demo/dev folder. The API discovers
manifests automatically, so registering a scene is all it takes to get it
onto the map.

    python scripts/register_scenes.py --source zenodo-gom

The Zenodo Gulf of Mexico scenes are real Sentinel-1 acquisitions over
documented spills (cross-referenced against NOAA reports), already in
Sigma0 dB. Running our own detector over them produces genuine detections
rather than another synthetic demo.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_scene_date(name: str) -> datetime | None:
    """Pull an acquisition date out of the filename.

    These files are named by date (2018_09_26, 20191015, 20200224_b). Without
    a date there is no wind lookup and no AIS window, so a scene we cannot
    date is skipped rather than given a plausible-looking guess.
    """
    stem = Path(name).stem
    digits = re.sub(r"[^0-9]", "", stem)
    for fmt, length in (("%Y%m%d", 8), ("%Y%m", 6)):
        if len(digits) >= length:
            try:
                return datetime.strptime(digits[:length], fmt).replace(
                    hour=12, tzinfo=timezone.utc
                )
            except ValueError:
                continue
    return None


def register_zenodo_gom(raw_dir: Path, out_dir: Path, config: str) -> int:
    import rasterio

    from ingest.raster import _bounds_to_bbox

    scenes: list[Path] = []
    for split in ("test", "train"):
        scenes.extend(sorted((raw_dir / split / "images").glob("*.tif")))
    if not scenes:
        raise SystemExit(f"No scenes under {raw_dir}. Run scripts/prepare_dataset.py first.")

    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0

    for path in scenes:
        acquired = parse_scene_date(path.name)
        if acquired is None:
            print(f"  skip {path.name}: no date in the filename")
            skipped += 1
            continue

        with rasterio.open(path) as src:
            bbox = _bounds_to_bbox(src.bounds, src.crs)
            height, width = src.height, src.width
        if bbox is None:
            print(f"  skip {path.name}: no usable geolocation")
            skipped += 1
            continue

        scene_id = f"GOM_{path.stem.strip('_').upper()}"
        manifest = {
            "scene_id": scene_id,
            "acquired_at": acquired.isoformat(),
            "bbox": [round(v, 6) for v in bbox],
            "vv_path": str(path.relative_to(REPO_ROOT)),
            "vh_path": None,          # this set is single-pol
            "orbit_direction": "DESCENDING",
            "config": config,
            "source": "Zenodo 4672426 (CC-BY-4.0), Gulf of Mexico oil spills 2018-2020",
            "REAL_IMAGERY": True,
            "note": (
                "Real Sentinel-1 acquisition over a documented Gulf of Mexico "
                "spill. Detections here come from our own pipeline."
            ),
            "size": [height, width],
        }
        (out_dir / f"{scene_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  {scene_id}  {acquired:%Y-%m-%d}  bbox {[round(v,2) for v in bbox]}  {width}x{height}")
        written += 1

    print(f"\nRegistered {written} real scene(s), skipped {skipped}. -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="zenodo-gom", choices=["zenodo-gom"])
    ap.add_argument("--raw-dir", default="data/raw/extracted")
    ap.add_argument("--out-dir", default="data/dev/scenes")
    ap.add_argument("--config", default="configs/real_gom.yaml")
    args = ap.parse_args()
    return register_zenodo_gom(REPO_ROOT / args.raw_dir, REPO_ROOT / args.out_dir, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
