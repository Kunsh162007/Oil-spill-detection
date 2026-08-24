"""Download Sentinel-1 GRD scenes from the Copernicus Data Space Ecosystem.

Run by teammates who do not read Python. Config-driven, no code edits.

    python scripts/fetch_sentinel.py --config configs/fetch_elsa3.yaml

Prints exactly what it got - scene count, IDs, dates, bytes - and exits
non-zero on an empty result. CDSE answers a slightly-wrong bbox with an empty
list rather than an error, and a silent empty download looks identical to
"there was no pass that day".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config, resolve_path  # noqa: E402

CATALOGUE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
DOWNLOAD = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"


def get_token(username: str, password: str) -> str:
    import requests

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(
            f"CDSE login failed ({resp.status_code}). Check CDSE_USERNAME and "
            f"CDSE_PASSWORD in .env - register free at dataspace.copernicus.eu"
        )
    return resp.json()["access_token"]


def search(bbox, start: datetime, end: datetime, product_type="GRD", limit=20):
    """Query the CDSE OData catalogue. bbox is (min_lon, min_lat, max_lon, max_lat)."""
    import requests

    min_lon, min_lat, max_lon, max_lat = bbox
    # OData wants a closed WKT ring, longitude first.
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
        CATALOGUE,
        params={"$filter": filt, "$top": limit, "$orderby": "ContentDate/Start asc"},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json().get("value", [])


def download(product_id: str, name: str, out_dir: Path, token: str) -> Path:
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.zip"
    if target.exists() and target.stat().st_size > 0:
        print(f"  already have {target.name} ({target.stat().st_size/1e6:.0f} MB)")
        return target

    print(f"  downloading {name} ...", flush=True)
    with requests.get(
        DOWNLOAD.format(pid=product_id),
        headers={"Authorization": f"Bearer {token}"},
        stream=True, timeout=1800, allow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        # Stream to a temp name so an interrupted run never leaves a file that
        # looks complete.
        tmp = target.with_suffix(".part")
        written = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if written % (100 << 20) < (1 << 20):
                    print(f"    {written/1e6:.0f} MB", flush=True)
        tmp.replace(target)
    print(f"  saved {target.name} ({written/1e6:.0f} MB)")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML config with a fetch section")
    ap.add_argument("--list-only", action="store_true", help="search without downloading")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    config = load_config(args.config)
    fetch = config.section("fetch")
    if not fetch:
        raise SystemExit(f"{args.config} has no 'fetch:' section")

    bbox = fetch.get("bbox")
    if not bbox or len(bbox) != 4:
        raise SystemExit("fetch.bbox must be [min_lon, min_lat, max_lon, max_lat]")
    bbox = tuple(float(v) for v in bbox)

    date = fetch.get("date")
    if not date:
        raise SystemExit("fetch.date is required (YYYY-MM-DD)")
    centre = datetime.fromisoformat(str(date)).replace(tzinfo=timezone.utc)
    window_days = float(fetch.get("window_days", 3))
    start, end = centre - timedelta(days=window_days), centre + timedelta(days=window_days)

    out_dir = resolve_path(fetch.get("out_dir", "data/dev"))

    print(f"Searching Sentinel-1 {fetch.get('product_type','GRD')}")
    print(f"  bbox   : {bbox}  (min_lon, min_lat, max_lon, max_lat)")
    print(f"  window : {start:%Y-%m-%d} to {end:%Y-%m-%d}")

    products = search(bbox, start, end, str(fetch.get("product_type", "GRD")),
                      int(fetch.get("limit", 20)))

    if not products:
        print("\nNO SCENES FOUND.", file=sys.stderr)
        print(
            "CDSE returns an empty list rather than an error when the bbox is\n"
            "slightly wrong or no pass exists. Check:\n"
            "  1. bbox order is min_lon, min_lat, max_lon, max_lat (LON FIRST)\n"
            "  2. the date window is wide enough (revisit is 6-12 days)\n"
            "  3. the area is actually covered by Sentinel-1",
            file=sys.stderr,
        )
        return 1

    print(f"\nFound {len(products)} scene(s):")
    for p in products:
        size_mb = float(p.get("ContentLength", 0)) / 1e6
        print(f"  {p['Name']}")
        print(f"    id={p['Id']}  start={p.get('ContentDate',{}).get('Start')}  {size_mb:.0f} MB")

    if args.list_only:
        print("\n--list-only: nothing downloaded.")
        return 0

    username = os.environ.get("CDSE_USERNAME", "")
    password = os.environ.get("CDSE_PASSWORD", "")
    if not (username and password):
        print(
            "\nCDSE_USERNAME / CDSE_PASSWORD are not set in .env - cannot download.\n"
            "Register free at https://dataspace.copernicus.eu, then copy\n"
            ".env.example to .env and fill them in.",
            file=sys.stderr,
        )
        return 2

    token = get_token(username, password)
    downloaded = []
    for p in products[: int(fetch.get("max_downloads", 3))]:
        downloaded.append(download(p["Id"], p["Name"], out_dir, token))

    index = out_dir / "fetch_index.json"
    index.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "bbox": list(bbox),
        "window": [start.isoformat(), end.isoformat()],
        "scenes": [
            {"name": p["Name"], "id": p["Id"],
             "start": p.get("ContentDate", {}).get("Start"),
             "bytes": p.get("ContentLength")}
            for p in products
        ],
    }, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in downloaded if f.exists())
    print(f"\nDownloaded {len(downloaded)} scene(s), {total/1e9:.2f} GB -> {out_dir}")
    print(f"Index written to {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
