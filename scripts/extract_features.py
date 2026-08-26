"""Build a look-alike feature table from the Yang et al. PANGAEA dataset.

    python scripts/extract_features.py --max-scenes 40
    python scripts/train.py --config configs/train_baseline.yaml --stage lookalike

What this uses, and what it cannot.

PANGAEA 10.1594/PANGAEA.980773 publishes two halves. The IMAGE files - JPG
patches with Pascal VOC boxes - are behind authorisation: the bulk archive
answers 401 and refers you to the principal investigator. The TABULAR metadata
is open, and it turns out to be the more useful half. For each of 5,515
annotated objects it gives the class, the acquisition time, the patch corners,
the object box in pixels and in degrees, and - decisively - the identifier of
the Sentinel-1 product the patch was cut from.

Those products are on the AWS Open Data mirror, unauthenticated. So instead of
deriving physics from 8-bit JPGs, this reads the ORIGINAL calibrated imagery
through the same windowed path the pipeline uses and computes features from
Sigma0 in dB. A coefficient fitted here is fitted in the space production runs
in, which a JPG-derived proxy would not be.

Limits that belong with any number quoted from this:

  * Annotations are bounding BOXES, not masks, so elongation and compactness
    are box-derived and coarser than the polygon values computed at inference.
    Directionally right, not identical.
  * The dataset publishes NO k-means cluster labels, whatever secondary
    descriptions claim. The only subgroup split available is coast vs water,
    so that is what false-positive rate gets reported against.
  * 1,181 distinct products cover the dataset. Reading all of them is
    impractical, so a class-balanced scene sample is taken and --max-scenes
    bounds it.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

METADATA = "data/raw/pangaea/pang.txt"

# Column positions in the PANGAEA tab-separated export, from its header row.
COL_SET, COL_START, COL_SENTINEL = 0, 5, 7
COL_PATCH_LON_UL, COL_PATCH_LAT_UL = 10, 11
COL_PATCH_LON_BR, COL_PATCH_LAT_BR = 14, 15
COL_OBJ_LON_UL, COL_OBJ_LAT_UL = 18, 19
COL_OBJ_LON_BR, COL_OBJ_LAT_BR = 22, 23

OIL_SETS = {"ow", "oc"}
COAST_SETS = {"oc", "nc"}


def parse_metadata(path: Path) -> list[dict]:
    """One record per annotated object, from the open tabular export."""
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not re.match(r"^(oc|ow|nc|nw)\t", line):
            continue
        parts = line.split("\t")
        if len(parts) < 24:
            continue
        try:
            patch = _bbox(parts, COL_PATCH_LON_UL, COL_PATCH_LAT_UL,
                          COL_PATCH_LON_BR, COL_PATCH_LAT_BR)
            if patch is None:
                continue
            rows.append({
                "set": parts[COL_SET],
                "sentinel_id": parts[COL_SENTINEL],
                "when": datetime.fromisoformat(
                    parts[COL_START]).replace(tzinfo=timezone.utc),
                "patch_bbox": patch,
                # A no-oil patch carries no annotated object - there is nothing
                # to box, the whole patch IS the look-alike. Falling back to the
                # patch extent is what makes the negative class usable at all;
                # requiring an object box silently dropped all 2,290 of them.
                "obj_bbox": _bbox(parts, COL_OBJ_LON_UL, COL_OBJ_LAT_UL,
                                  COL_OBJ_LON_BR, COL_OBJ_LAT_BR) or patch,
            })
        except (ValueError, IndexError):
            continue
    return rows


def _bbox(parts, lon_a, lat_a, lon_b, lat_b):
    """min_lon, min_lat, max_lon, max_lat - longitude first, as everywhere.

    Returns None when the columns are blank, which is how the export records a
    patch with no annotated object rather than an error.
    """
    try:
        lons = (float(parts[lon_a]), float(parts[lon_b]))
        lats = (float(parts[lat_a]), float(parts[lat_b]))
    except (ValueError, IndexError):
        return None
    return (min(lons), min(lats), max(lons), max(lats))


def geometry_features(obj_bbox) -> dict[str, float]:
    """Shape from the annotation box.

    Compactness uses the same 4*pi*A/P^2 definition as polygonize, so the value
    stays on one scale - though a rectangle cannot reach a polygon's extremes.
    """
    from core.geo import haversine_km

    min_lon, min_lat, max_lon, max_lat = obj_bbox
    width_km = max(haversine_km((min_lon, min_lat), (max_lon, min_lat)), 1e-4)
    height_km = max(haversine_km((min_lon, min_lat), (min_lon, max_lat)), 1e-4)

    area = width_km * height_km
    perimeter = 2.0 * (width_km + height_km)
    return {
        "elongation": max(width_km, height_km) / min(width_km, height_km),
        "compactness": (4.0 * math.pi * area) / (perimeter ** 2),
        "area_km2": area,
        "log_area": math.log10(max(area, 1e-6)),
    }


def radiometry_features(vv_db, vh_db, window_px) -> dict[str, float]:
    """Damping and texture from calibrated Sigma0, as the pipeline computes them.

    Damping is the surrounding sea's median dB minus the object's median dB -
    the definition used at inference, which is what makes a fitted coefficient
    transferable rather than merely plausible.
    """
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops

    r0, c0, r1, c1 = window_px
    inside = vv_db[r0:r1, c0:c1]
    finite_inside = inside[np.isfinite(inside)]
    if finite_inside.size < 16:
        raise ValueError("object window has too few finite samples")

    pad_r, pad_c = max(4, (r1 - r0) // 2), max(4, (c1 - c0) // 2)
    top, left = max(0, r0 - pad_r), max(0, c0 - pad_c)
    ring = vv_db[top:r1 + pad_r, left:c1 + pad_c].astype(float).copy()
    ring[r0 - top:r0 - top + (r1 - r0), c0 - left:c0 - left + (c1 - c0)] = np.nan
    ring = ring[np.isfinite(ring)]
    if ring.size < 16:
        raise ValueError("no surrounding sea to compare against")

    lo, hi = np.percentile(finite_inside, [2, 98])
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise ValueError("degenerate dynamic range")
    scaled = np.clip((np.nan_to_num(inside, nan=float(lo)) - lo) / (hi - lo), 0, 1)
    levels = 32
    glcm = graycomatrix((scaled * (levels - 1)).astype(np.uint8), distances=[1],
                        angles=[0, np.pi / 2], levels=levels,
                        symmetric=True, normed=True)

    out = {
        "damping_db": float(np.median(ring) - np.median(finite_inside)),
        "texture_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "texture_contrast": float(graycoprops(glcm, "contrast").mean()),
        "has_vh": 0.0,
        "vh_vv_ratio": float("nan"),
    }

    if vh_db is not None and vh_db.shape == vv_db.shape:
        vh_inside = vh_db[r0:r1, c0:c1]
        vh_inside = vh_inside[np.isfinite(vh_inside)]
        if vh_inside.size >= 16:
            # Ratio in linear power, matching the pipeline's convention.
            vv_lin = 10.0 ** (float(np.median(finite_inside)) / 10.0)
            vh_lin = 10.0 ** (float(np.median(vh_inside)) / 10.0)
            if vv_lin > 0:
                out["has_vh"] = 1.0
                out["vh_vv_ratio"] = vh_lin / vv_lin
    return out


FIELDNAMES = ["cluster", "wind_window_score", "wind_speed_ms", "damping_db", "elongation",
              "compactness", "log_area", "area_km2", "texture_homogeneity",
              "texture_contrast", "has_vh", "vh_vv_ratio",
              "label", "subset", "set", "sentinel_id"]


def extract_scene(product, objects, wind_lookup, writer) -> int:
    """Read one product once and pull every sampled object out of it."""
    import numpy as np

    from scripts.fetch_aws import BUCKET, aws_prefix, read_window

    prefix = aws_prefix(product)
    if not prefix:
        raise ValueError("product name did not map to an AWS prefix")

    lons = [o["patch_bbox"][0] for o in objects] + [o["patch_bbox"][2] for o in objects]
    lats = [o["patch_bbox"][1] for o in objects] + [o["patch_bbox"][3] for o in objects]
    bbox = (min(lons) - 0.02, min(lats) - 0.02, max(lons) + 0.02, max(lats) + 0.02)

    from ingest.calibrate import to_sigma0_db

    # read_window returns "dB OR DN" - for these AWS GRDs it is raw digital
    # numbers, and a first run fitted on them produced damping from -101 to
    # +112 dB against a physical range of roughly 0-15. Everything downstream
    # here is defined in decibels, so calibrate exactly as the pipeline does
    # rather than assuming the units.
    vv_raw, actual = read_window(f"{BUCKET}/{prefix}/measurement/iw-vv.tiff", bbox)
    vv = to_sigma0_db(vv_raw).sigma0_db
    try:
        vh_raw, _ = read_window(f"{BUCKET}/{prefix}/measurement/iw-vh.tiff", bbox)
        vh = to_sigma0_db(vh_raw, polarisation="vh").sigma0_db
    except Exception:                                      # noqa: BLE001
        vh = None                                          # single-pol: degrade

    height, width = vv.shape
    span_lon = max(actual[2] - actual[0], 1e-9)
    span_lat = max(actual[3] - actual[1], 1e-9)
    written = 0

    for obj in objects:
        try:
            min_lon, min_lat, max_lon, max_lat = obj["obj_bbox"]
            c0 = int((min_lon - actual[0]) / span_lon * width)
            c1 = int((max_lon - actual[0]) / span_lon * width)
            # Rows run north to south, so latitude inverts.
            r0 = int((1.0 - (max_lat - actual[1]) / span_lat) * height)
            r1 = int((1.0 - (min_lat - actual[1]) / span_lat) * height)
            r0, r1 = max(0, min(r0, r1)), min(height, max(r0, r1))
            c0, c1 = max(0, min(c0, c1)), min(width, max(c0, c1))
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue

            geom = geometry_features(obj["obj_bbox"])
            radio = radiometry_features(vv, vh, (r0, c0, r1, c1))
            wind = wind_lookup((min_lon + max_lon) / 2.0,
                               (min_lat + max_lat) / 2.0, obj["when"])

            row = {
                "wind_window_score": round(wind.window_score, 4),
                "wind_speed_ms": round(wind.speed_ms, 3),
                "label": 1 if obj["set"] in OIL_SETS else 0,
                "subset": "coast" if obj["set"] in COAST_SETS else "water",
                # The subgroup false-positive rate is reported against. Yang
                # publishes only coast/water, so the low-wind slice - the one
                # that actually breaks detectors - is constructed here from
                # real ERA5, which every patch supports because it carries a
                # timestamp and a position.
                "cluster": ("low-wind-" if wind.window_score < 0.25 else "")
                           + ("coast" if obj["set"] in COAST_SETS else "water"),
                "set": obj["set"],
                "sentinel_id": product,
                "vh_vv_ratio": ("" if not np.isfinite(radio["vh_vv_ratio"])
                                else round(radio["vh_vv_ratio"], 5)),
            }
            for key in ("damping_db", "texture_homogeneity", "texture_contrast"):
                row[key] = round(radio[key], 5)
            row["has_vh"] = radio["has_vh"]
            for key in ("elongation", "compactness", "log_area", "area_km2"):
                row[key] = round(geom[key], 5)

            writer.writerow(row)
            written += 1
        except Exception as exc:                           # noqa: BLE001
            print(f"    object skipped: {type(exc).__name__}: {str(exc)[:80]}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default=METADATA)
    ap.add_argument("--out", default="data/dev/lookalike_features.csv")
    ap.add_argument("--max-scenes", type=int, default=40,
                    help="how many source products to read (1181 exist)")
    ap.add_argument("--max-objects-per-scene", type=int, default=12)
    args = ap.parse_args()

    from core.config import resolve_path
    from detect.wind import OpenMeteoERA5Wind

    meta_path = resolve_path(args.metadata)
    if not meta_path.exists():
        print(f"No PANGAEA metadata at {meta_path}. Fetch it with:\n"
              f'  curl -s "https://doi.pangaea.de/10.1594/PANGAEA.980773?format=textfile"'
              f" -o {args.metadata}", file=sys.stderr)
        return 1

    rows = parse_metadata(meta_path)
    if not rows:
        print(f"{meta_path} produced no usable rows", file=sys.stderr)
        return 1
    print(f"metadata: {len(rows)} annotated object(s)")

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_scene[row["sentinel_id"]].append(row)

    # The export is grouped by class, so a sample in file order would be almost
    # entirely oil. Balance it at the scene level instead.
    oil = [s for s, rs in by_scene.items() if any(r["set"] in OIL_SETS for r in rs)]
    look = [s for s, rs in by_scene.items()
            if all(r["set"] not in OIL_SETS for r in rs)]
    half = max(1, args.max_scenes // 2)
    selected = oil[:half] + look[:half]
    print(f"products: {len(by_scene)} total, sampling {len(selected)} "
          f"({len(oil[:half])} containing oil, {len(look[:half])} look-alike only)")

    wind_lookup = OpenMeteoERA5Wind()
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for n, scene_id in enumerate(selected, 1):
            objects = by_scene[scene_id][:args.max_objects_per_scene]
            print(f"[{n}/{len(selected)}] {scene_id[:50]}  {len(objects)} object(s)",
                  flush=True)
            try:
                written += extract_scene(scene_id.replace(".SAFE", ""), objects,
                                         wind_lookup, writer)
            except Exception as exc:                       # noqa: BLE001
                print(f"    scene skipped: {type(exc).__name__}: {str(exc)[:110]}")
                skipped += len(objects)

    if written == 0:
        print(f"\nFAILED: no features extracted; {skipped} object(s) skipped.",
              file=sys.stderr)
        return 1
    print(f"\nWrote {written} row(s), skipped {skipped} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
