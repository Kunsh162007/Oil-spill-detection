"""Fetch Sentinel-1 GRD from AWS Open Data - NO ACCOUNT REQUIRED.

    python scripts/fetch_aws.py --bbox 68,8,78,20 --days 3

The Copernicus Data Space Ecosystem needs a free account to download, which is
a barrier when its registration page will not render. This path avoids it
entirely:

  * SEARCH   the CDSE OData catalogue - open, no credentials
  * DOWNLOAD from the AWS Open Data mirror - open, no credentials

Scenes are NOT downloaded whole. A Sentinel-1 IW GRD measurement band is
several hundred megabytes, and we only need the part covering the area of
interest at ~80 m. GDAL reads that window straight out of the remote GeoTIFF
over HTTP, so a scene costs tens of megabytes instead of a gigabyte.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BUCKET = "https://sentinel-s1-l1c.s3.amazonaws.com"
NAME_RE = re.compile(
    r"^(?P<mission>S1[A-D])_(?P<mode>[A-Z0-9]{2})_(?P<ptype>GRD[HM])_"
    r"1(?P<class>[SA])(?P<pol>[DS][VH])_(?P<start>\d{8}T\d{6})"
)


def aws_prefix(product_name: str) -> str | None:
    """Map a Sentinel-1 product name onto its AWS Open Data prefix.

    Layout is GRD/{year}/{month}/{day}/{mode}/{pol}/{product}/ with month and
    day unpadded, which is easy to get wrong and yields a silent 404.
    """
    stem = product_name.replace(".SAFE", "")
    m = NAME_RE.match(stem)
    if not m:
        return None
    start = datetime.strptime(m.group("start"), "%Y%m%dT%H%M%S")
    return (
        f"GRD/{start.year}/{start.month}/{start.day}/"
        f"{m.group('mode')}/{m.group('pol')}/{stem}"
    )


def exists(url: str, timeout: float = 30.0) -> bool:
    import requests

    try:
        return requests.head(url, timeout=timeout, allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False


def search_cdse(bbox, start: datetime, end: datetime, limit: int = 40,
                mode: str = "IW", product_type: str = "GRD") -> list[dict]:
    """Search the open CDSE catalogue. No credentials needed for search."""
    import requests

    min_lon, min_lat, max_lon, max_lat = bbox
    polygon = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    filt = (
        f"Collection/Name eq 'SENTINEL-1' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') "
        f"and ContentDate/Start gt {start.strftime('%Y-%m-%dT%H:%M:%S.000Z')} "
        f"and ContentDate/Start lt {end.strftime('%Y-%m-%dT%H:%M:%S.000Z')} "
        f"and contains(Name,'{product_type}')"
    )
    resp = requests.get(
        "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
        params={"$filter": filt, "$top": limit, "$orderby": "ContentDate/Start desc"},
        timeout=90,
    )
    resp.raise_for_status()
    products = resp.json().get("value", [])
    return [p for p in products if f"_{mode}_" in p.get("Name", "")]


def read_window(url: str, bbox, target_res_m: float = 80.0):
    """Read the bbox subset of a remote GRD, decimated, without downloading it.

    GRD is in ground range with GCPs rather than a plain affine transform, so
    the bbox is mapped through those. Returns (array_dB_or_DN, actual_bbox).
    """
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(f"/vsicurl/{url}") as src:
        gcps, gcp_crs = src.gcps
        if not gcps:
            raise ValueError("no GCPs on this product - cannot geolocate the window")

        lons = [g.x for g in gcps]
        lats = [g.y for g in gcps]
        scene_bbox = (min(lons), min(lats), max(lons), max(lats))

        # Reject a non-overlapping request rather than returning an arbitrary
        # corner of the scene, which would look like a valid read.
        if (bbox[2] < scene_bbox[0] or bbox[0] > scene_bbox[2]
                or bbox[3] < scene_bbox[1] or bbox[1] > scene_bbox[3]):
            raise ValueError(f"requested bbox {bbox} does not overlap scene {scene_bbox}")

        from rasterio.transform import from_gcps

        transform = from_gcps(gcps)
        inv = ~transform
        corners = [
            inv * (bbox[0], bbox[1]), inv * (bbox[2], bbox[1]),
            inv * (bbox[2], bbox[3]), inv * (bbox[0], bbox[3]),
        ]
        cols = [c[0] for c in corners]
        rows = [c[1] for c in corners]
        col_off = max(0, int(min(cols)))
        row_off = max(0, int(min(rows)))
        width = min(src.width - col_off, int(max(cols) - min(cols)) + 1)
        height = min(src.height - row_off, int(max(rows) - min(rows)) + 1)
        if width <= 0 or height <= 0:
            raise ValueError("computed window is empty")

        # Decimate on read: 10 m native down to the target, so the transfer is
        # a fraction of the full band.
        factor = max(1, int(round(target_res_m / 10.0)))
        out_shape = (max(1, height // factor), max(1, width // factor))

        window = rasterio.windows.Window(col_off, row_off, width, height)
        data = src.read(1, window=window, out_shape=out_shape,
                        resampling=rasterio.enums.Resampling.average)

        top_left = transform * (col_off, row_off)
        bottom_right = transform * (col_off + width, row_off + height)
        actual = (
            min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1]),
            max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1]),
        )
    return np.asarray(data), actual


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bbox", default="68,8,78,20",
                    help="min_lon,min_lat,max_lon,max_lat (default: Arabian Sea / Indian coast)")
    ap.add_argument("--days", type=float, default=3.0, help="how far back to search")
    ap.add_argument("--date", default=None,
                    help="centre the search on this date (YYYY-MM-DD) instead of now, "
                         "for archived incidents")
    ap.add_argument("--name", default=None, help="label prefix for the fetched scenes")
    ap.add_argument("--max-scenes", type=int, default=3)
    ap.add_argument("--out-dir", default="data/live")
    ap.add_argument("--config", default="configs/live.yaml")
    ap.add_argument("--resolution", type=float, default=80.0)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox needs 4 comma-separated values")

    if args.date:
        centre = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
        start = centre - timedelta(days=args.days)
        end = centre + timedelta(days=args.days)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
    print(f"Searching Sentinel-1 IW GRD  (no account needed)")
    print(f"  bbox   : {bbox}")
    print(f"  window : {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")

    products = search_cdse(bbox, start, end)
    if not products:
        print("\nNO SCENES FOUND in that window.", file=sys.stderr)
        print("Sentinel-1 revisit is 6-12 days per point, so widen --days or "
              "the bbox.", file=sys.stderr)
        return 1

    print(f"\nCatalogue returned {len(products)} IW scene(s):")
    usable = []
    for p in products:
        name = p["Name"]
        prefix = aws_prefix(name)
        acquired = p.get("ContentDate", {}).get("Start", "")
        if prefix is None:
            print(f"  - {name[:62]}  (unparseable name)")
            continue
        vv_url = f"{BUCKET}/{prefix}/measurement/iw-vv.tiff"
        age_h = (end - datetime.fromisoformat(acquired.replace("Z", "+00:00"))).total_seconds() / 3600
        on_aws = exists(vv_url)
        print(f"  {'OK ' if on_aws else '-- '} {name[:58]}")
        print(f"      {acquired[:16]}  age {age_h:.0f} h  "
              f"{'on AWS mirror' if on_aws else 'not yet mirrored to AWS'}")
        if on_aws:
            usable.append((name, prefix, acquired, age_h))
        if len(usable) >= args.max_scenes:
            break

    if not usable:
        print("\nNone of these are on the AWS mirror yet. The mirror lags the "
              "catalogue by a few hours; try --days 4.", file=sys.stderr)
        return 1

    if args.list_only:
        print(f"\n--list-only: {len(usable)} scene(s) available, nothing fetched.")
        return 0

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from ingest.raster import write_raster

    written = 0
    for name, prefix, acquired, age_h in usable:
        print(f"\nReading {name[:58]}")
        try:
            vv, actual_bbox = read_window(
                f"{BUCKET}/{prefix}/measurement/iw-vv.tiff", bbox, args.resolution)
        except Exception as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        scene_id = name.replace(".SAFE", "")
        vv_path = write_raster(out_dir / f"{scene_id}_vv.tif", vv.astype("float32"), actual_bbox)
        print(f"  VV {vv.shape} -> {vv_path.name} ({vv_path.stat().st_size/1e6:.1f} MB)")

        vh_path = None
        try:
            vh, _ = read_window(
                f"{BUCKET}/{prefix}/measurement/iw-vh.tiff", bbox, args.resolution)
            if vh.shape == vv.shape:
                vh_path = write_raster(out_dir / f"{scene_id}_vh.tif",
                                       vh.astype("float32"), actual_bbox)
                print(f"  VH {vh.shape} -> {vh_path.name}")
        except Exception as exc:
            print(f"  VH unavailable ({exc}); continuing single-pol")

        manifest = {
            "scene_id": scene_id,
            "label": args.name,
            "acquired_at": datetime.fromisoformat(
                acquired.replace("Z", "+00:00")).isoformat(),
            "bbox": [round(v, 6) for v in actual_bbox],
            "vv_path": str(vv_path.relative_to(REPO_ROOT)),
            "vh_path": str(vh_path.relative_to(REPO_ROOT)) if vh_path else None,
            "orbit_direction": "UNKNOWN",
            "config": args.config,
            "REAL_IMAGERY": True,
            "source": "AWS Open Data sentinel-s1-l1c (public, no credentials)",
            "age_hours_at_fetch": round(age_h, 1),
        }
        (out_dir / f"{scene_id}.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
        written += 1

    print(f"\nFetched {written} scene(s) -> {out_dir}")
    print("Restart the API to pick them up.")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
