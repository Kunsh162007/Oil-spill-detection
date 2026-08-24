"""Download the documented spill registries that back the world map.

    python scripts/fetch_incidents.py

Unlike the detectors, these are CONFIRMED events - spills that are known to
have happened, from public incident registries. They serve two purposes:

  1. a genuinely worldwide map layer of real oil spills
  2. independent corroboration for our own SAR detections

Sources:
  * NOAA IncidentNews - ~5,000 responded-to incidents since 1957, open CSV
  * a curated catalogue of major world spills, compiled from public record,
    which fills in the coverage NOAA's US focus leaves out
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

NOAA_CSV = "https://incidentnews.noaa.gov/raw/incidents.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/reference")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    import requests

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "noaa_incidents.csv"

    if csv_path.exists() and not args.force:
        print(f"Already have {csv_path} ({csv_path.stat().st_size/1e6:.1f} MB); "
              f"pass --force to refresh.")
    else:
        print(f"Downloading NOAA IncidentNews -> {csv_path}")
        resp = requests.get(NOAA_CSV, timeout=180)
        resp.raise_for_status()
        if len(resp.content) < 10_000:
            print("NOAA returned a suspiciously small file - refusing to overwrite.",
                  file=sys.stderr)
            return 1
        csv_path.write_bytes(resp.content)
        print(f"  {len(resp.content)/1e6:.1f} MB")

    from detect.incidents import load_registry

    registry = load_registry(csv_path)
    petroleum = [i for i in registry.incidents if i.is_petroleum]
    with_dates = [i for i in petroleum if i.occurred_at]
    sentinel_era = [i for i in with_dates if i.occurred_at.year >= 2014]

    lons = [i.lon for i in petroleum]
    lats = [i.lat for i in petroleum]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_incidents": len(registry),
        "petroleum_incidents": len(petroleum),
        "sentinel1_era": len(sentinel_era),
        "bbox": [round(min(lons), 3), round(min(lats), 3),
                 round(max(lons), 3), round(max(lats), 3)],
        "sources": sorted({i.source for i in registry.incidents}),
    }
    (out_dir / "incidents_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nRegistry: {len(registry)} documented spills")
    print(f"  petroleum only    : {len(petroleum)}")
    print(f"  Sentinel-1 era    : {len(sentinel_era)} (since 2014)")
    print(f"  coverage          : lon {summary['bbox'][0]} to {summary['bbox'][2]}, "
          f"lat {summary['bbox'][1]} to {summary['bbox'][3]}")
    print(f"  sources           : {', '.join(summary['sources'])}")
    print(f"\nSummary -> {out_dir/'incidents_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
