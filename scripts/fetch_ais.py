"""Download AIS vessel tracks.

    python scripts/fetch_ais.py --config configs/fetch_elsa3.yaml

Three sources:
  * noaa    - NOAA Marine Cadastre, open HTTP, no account. US waters, 2015
              onward, ~330 MB/day. The only free source here that publishes
              real POSITION reports, which is what the scoring needs.
              https://coast.noaa.gov/htdata/CMSP/AISDataHandler/
  * danish  - Danish Maritime Authority, open HTTP, no auth. ~2 GB/day, so
              it is filtered to the bbox on the fly rather than kept whole.
              (Host web.ais.dk was unreachable as of Aug 2026.)
  * gfw     - Global Fishing Watch, free non-commercial token, covers Indian
              waters, lags about 72 h. Identity and EVENTS only - it does not
              serve position tracks, so it cannot drive parity, proximity or
              temporality. Use it for AIS-gap dark-vessel signals.

Exits non-zero on an empty result. These feeds return nothing rather than an
error when a bbox is slightly wrong, and "no ships were there" is a very
different claim from "the query was malformed".
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config, resolve_path  # noqa: E402

DANISH_URL = "http://web.ais.dk/aisdata/aisdk-{date}.zip"


def fetch_danish(day: datetime, bbox, out_path: Path, max_rows: int = 4_000_000) -> int:
    """Stream one day of Danish AIS, keeping only rows inside the bbox.

    The daily file is ~2 GB. We never write it to disk whole - it is filtered
    during the download, which keeps the working set inside the 50 GB target.
    """
    import requests

    min_lon, min_lat, max_lon, max_lat = bbox
    url = DANISH_URL.format(date=day.strftime("%Y-%m-%d"))
    print(f"Streaming {url}")
    print(f"  filtering to bbox {bbox}")

    resp = requests.get(url, stream=True, timeout=600)
    if resp.status_code == 404:
        print(f"\nNo Danish AIS file for {day:%Y-%m-%d}. Files exist for roughly "
              f"the last two years.", file=sys.stderr)
        return 0
    resp.raise_for_status()

    buffer = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buffer.write(chunk)
        downloaded += len(chunk)
        if downloaded % (200 << 20) < (1 << 20):
            print(f"    {downloaded/1e6:.0f} MB", flush=True)
    buffer.seek(0)

    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buffer) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw, out_path.open("w", newline="", encoding="utf-8") as out:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
            writer = None
            for row in reader:
                try:
                    lat = float(row.get("Latitude", "nan"))
                    lon = float(row.get("Longitude", "nan"))
                except (TypeError, ValueError):
                    continue
                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                kept += 1
                if kept >= max_rows:
                    print(f"  hit the {max_rows} row cap; stopping")
                    break
    return kept


NOAA_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{date}.zip"

# Marine Cadastre column names, checked rather than assumed: a renamed column
# would yield an empty file, which reads as "no ships were there".
NOAA_REQUIRED = ("MMSI", "BaseDateTime", "LAT", "LON")


def fetch_noaa(days, bbox, start: datetime, end: datetime, out_path: Path) -> int:
    """Stream NOAA Marine Cadastre days, keeping rows inside bbox and window.

    A national day is 300-360 MB zipped and several GB unpacked, so nothing is
    unpacked and nothing is held in memory: the archive goes to a temporary
    file, the CSV member is read line by line straight out of it, and the
    archive is deleted. What lands on disk is a few hundred KB for one scene.

    Rows are written through with their original column names - load_ais_csv
    already understands MMSI / BaseDateTime / LAT / LON.
    """
    import requests

    min_lon, min_lat, max_lon, max_lat = bbox
    kept = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as out:
        writer = None
        for day in days:
            url = NOAA_URL.format(year=day.strftime("%Y"), date=day.strftime("%Y_%m_%d"))
            print(f"Streaming {url}", flush=True)
            # Deliberately not a `with` block: Windows refuses to unlink a file
            # that still has an open handle, and zipfile reopens it by path.
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            tmp_path = Path(tmp.name)
            try:
                with requests.get(url, stream=True, timeout=600) as resp:
                    if resp.status_code == 404:
                        print(f"  not published for {day:%Y-%m-%d}", file=sys.stderr)
                        continue
                    resp.raise_for_status()
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        tmp.write(chunk)
                        downloaded += len(chunk)
                tmp.close()
                print(f"  {downloaded/1e6:.0f} MB, scanning without unpacking", flush=True)
                kept, writer = _scan_noaa_zip(
                    tmp_path, url, bbox, start, end, out, writer, kept)
            finally:
                try:
                    tmp.close()
                except OSError:
                    pass
                tmp_path.unlink(missing_ok=True)
            print(f"  kept {kept:,} row(s) so far", flush=True)
    return kept


def _scan_noaa_zip(zip_path: Path, url: str, bbox, start: datetime, end: datetime,
                   out, writer, kept: int):
    """Filter one archive's CSV member into `out`. Returns (kept, writer)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"no CSV inside {url}")
        with zf.open(members[0]) as raw:
            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            missing = [c for c in NOAA_REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(
                    f"{members[0]} is missing {missing}; columns are "
                    f"{reader.fieldnames}. The Marine Cadastre schema changed - "
                    f"update NOAA_REQUIRED.")
            for row in reader:
                try:
                    lon = float(row["LON"])
                    lat = float(row["LAT"])
                except (TypeError, ValueError):
                    continue
                if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                    continue
                try:
                    when = datetime.fromisoformat(row["BaseDateTime"])
                except (TypeError, ValueError):
                    continue
                if not (start <= when <= end):
                    continue
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                    writer.writeheader()
                writer.writerow(row)
                kept += 1
    return kept, writer


def fetch_gfw(bbox, start: datetime, end: datetime, out_path: Path) -> int:
    from attribute.ais import GFWClient

    token = os.environ.get("GFW_TOKEN", "")
    if not token:
        print("GFW_TOKEN is not set in .env. Get a free non-commercial token at\n"
              "globalfishingwatch.org/our-apis/", file=sys.stderr)
        return 0

    tracks = GFWClient(token).fetch_tracks(bbox, start, end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["MMSI", "Name", "Ship type", "Flag"])
        for t in tracks:
            writer.writerow([t.mmsi, t.name or "", t.vessel_type or "", t.flag or ""])
    return len(tracks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", choices=["noaa", "danish", "gfw"], default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    config = load_config(args.config)
    fetch = config.section("fetch")
    bbox = tuple(float(v) for v in fetch["bbox"])
    centre = datetime.fromisoformat(str(fetch["date"])).replace(tzinfo=timezone.utc)
    source = args.source or str(fetch.get("ais_source", "danish"))
    out_path = resolve_path(fetch.get("ais_path", f"data/dev/ais_{centre:%Y%m%d}.csv"))

    if source == "noaa":
        # The attribution window: CLAUDE.md asks for 8 h before the pass to 6 h
        # after, which usually straddles midnight and so spans two daily files.
        window_start = centre - timedelta(hours=float(fetch.get("ais_hours_before", 8)))
        window_end = centre + timedelta(hours=float(fetch.get("ais_hours_after", 6)))
        days, cursor = [], window_start.date()
        while cursor <= window_end.date():
            days.append(datetime(cursor.year, cursor.month, cursor.day))
            cursor = cursor.fromordinal(cursor.toordinal() + 1)
        print(f"NOAA Marine Cadastre  window {window_start} to {window_end} UTC "
              f"({len(days)} daily file(s))")
        kept = fetch_noaa(days, bbox, window_start.replace(tzinfo=None),
                          window_end.replace(tzinfo=None), out_path)
    elif source == "danish":
        # The Danish feed covers Danish waters; it is the development source,
        # not the demo source. Warn rather than silently return nothing.
        if not (3.0 <= bbox[0] <= 16.0 and 53.0 <= bbox[1] <= 60.0):
            print(
                f"WARNING: bbox {bbox} is outside Danish waters. The DMA feed\n"
                f"covers roughly lon 3-16, lat 53-60. Use --source gfw for\n"
                f"Indian waters.", file=sys.stderr,
            )
        kept = fetch_danish(centre, bbox, out_path)
    else:
        kept = fetch_gfw(bbox, centre - timedelta(days=1), centre + timedelta(days=1), out_path)

    if kept == 0:
        print("\nNO AIS RECORDS RETURNED.", file=sys.stderr)
        print(
            "These feeds return empty rather than erroring. Check:\n"
            "  1. bbox order is min_lon, min_lat, max_lon, max_lat (LON FIRST)\n"
            "  2. the source actually covers that region\n"
            "  3. free AIS lags about 72 h - very recent dates may be absent",
            file=sys.stderr,
        )
        return 1

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
    print(f"\nWrote {kept} AIS rows -> {out_path} ({size_mb:.1f} MB)")

    try:
        from attribute.ais import ais_data_age_hours, load_ais_csv

        tracks = load_ais_csv(out_path, source=source)
        age = ais_data_age_hours(tracks)
        print(f"  vessels  : {len(tracks)}")
        if age is not None:
            print(f"  data age : {age:.1f} h behind now "
                  f"(free AIS lags ~72 h - never claim real-time)")
    except Exception as exc:
        print(f"  (could not summarise: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
