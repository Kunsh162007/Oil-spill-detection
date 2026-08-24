"""Download ERA5 10 m wind from the Copernicus Climate Data Store.

    python scripts/fetch_wind.py --config configs/fetch_elsa3.yaml

Wind is the strongest single feature for rejecting look-alikes, and
CLAUDE.md rule 3 says a candidate without wind context is not a candidate.
So this script never silently produces a partial file: it prints the exact
coverage it downloaded and exits non-zero if the request came back empty.

Needs ~/.cdsapirc. Register free at cds.climate.copernicus.eu and accept the
ERA5 licence, or the request fails with a licence error that reads like a
permissions bug.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config, resolve_path  # noqa: E402

DATASET = "reanalysis-era5-single-levels"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="override the output .nc path")
    args = ap.parse_args()

    config = load_config(args.config)
    fetch = config.section("fetch")
    bbox = fetch.get("bbox")
    if not bbox or len(bbox) != 4:
        raise SystemExit("fetch.bbox must be [min_lon, min_lat, max_lon, max_lat]")

    date = fetch.get("date")
    if not date:
        raise SystemExit("fetch.date is required (YYYY-MM-DD)")
    centre = datetime.fromisoformat(str(date)).replace(tzinfo=timezone.utc)
    pad_days = int(fetch.get("wind_pad_days", 1))
    days = [centre + timedelta(days=d) for d in range(-pad_days, pad_days + 1)]

    out_path = Path(args.out) if args.out else resolve_path(
        fetch.get("wind_path", f"data/dev/era5_{centre:%Y%m%d}.nc")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Already have {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return 0

    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    # CDS wants area as [North, West, South, East] - a different order from
    # our bbox, and getting it wrong silently returns the wrong ocean.
    area = [max_lat + 0.5, min_lon - 0.5, min_lat - 0.5, max_lon + 0.5]

    try:
        import cdsapi
    except ImportError:
        raise SystemExit("cdsapi is not installed. pip install cdsapi")

    print(f"Requesting ERA5 10 m wind")
    print(f"  area (N,W,S,E): {area}")
    print(f"  days          : {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}")
    print("  This can queue for several minutes at CDS.", flush=True)

    client = cdsapi.Client()
    try:
        client.retrieve(
            DATASET,
            {
                "product_type": "reanalysis",
                "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
                "year": sorted({f"{d:%Y}" for d in days}),
                "month": sorted({f"{d:%m}" for d in days}),
                "day": sorted({f"{d:%d}" for d in days}),
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area,
                "format": "netcdf",
            },
            str(out_path),
        )
    except Exception as exc:
        print(f"\nERA5 request failed: {exc}", file=sys.stderr)
        print(
            "Common causes:\n"
            "  1. ~/.cdsapirc missing or wrong (see cds.climate.copernicus.eu/api-how-to)\n"
            "  2. the ERA5 licence has not been accepted on the CDS website\n"
            "  3. the area is malformed - CDS wants [North, West, South, East]",
            file=sys.stderr,
        )
        return 1

    if not out_path.exists() or out_path.stat().st_size == 0:
        print("ERA5 returned an EMPTY file - refusing to continue.", file=sys.stderr)
        return 1

    print(f"\nSaved {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    try:
        import xarray as xr

        with xr.open_dataset(out_path) as ds:
            print(f"  variables : {list(ds.data_vars)}")
            print(f"  dimensions: {dict(ds.sizes)}")
            if "u10" in ds and "v10" in ds:
                import numpy as np

                speed = np.hypot(ds["u10"].values, ds["v10"].values)
                print(f"  wind speed: {np.nanmin(speed):.1f} - {np.nanmax(speed):.1f} m/s "
                      f"(mean {np.nanmean(speed):.1f})")
                print(f"  NOTE: SAR oil detection works roughly 3-10 m/s.")
    except Exception as exc:
        print(f"  (could not summarise the file: {exc})")

    print(f"\nSet wind.source: era5 and wind.path: {out_path} in your config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
