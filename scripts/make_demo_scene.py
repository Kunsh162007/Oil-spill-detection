"""Generate a synthetic Sentinel-1-like scene for testing and demos.

Produces a GeoTIFF plus a scene manifest containing a known bilge-dump
streak, a calm-wind look-alike and an algal bloom, so the whole pipeline can
be exercised without a 1 GB download and without network access.

Everything it writes is tagged synthetic. It exists so the pipeline can be
tested, NOT to stand in for real data in any reported number.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def gamma_speckle(shape, enl: float, rng) -> np.ndarray:
    """Multiplicative speckle with the right statistics for a multi-look GRD."""
    return rng.gamma(shape=enl, scale=1.0 / enl, size=shape)


def draw_streak(sigma0, start_rc, end_rc, width_px, damping, taper=True):
    """Paint a tapering linear slick - the signature of a moving vessel."""
    h, w = sigma0.shape
    r0, c0 = start_rc
    r1, c1 = end_rc
    length = int(math.hypot(r1 - r0, c1 - c0))
    for i in range(length):
        t = i / max(length - 1, 1)
        r = r0 + (r1 - r0) * t
        c = c0 + (c1 - c0) * t
        # A discharge is widest and darkest at the fresh end, fading behind.
        half = width_px * (1.0 - 0.55 * t) / 2.0 if taper else width_px / 2.0
        strength = damping * (1.0 - 0.35 * t) if taper else damping
        rr = np.arange(max(0, int(r - half)), min(h, int(r + half) + 1))
        cc = np.arange(max(0, int(c - half)), min(w, int(c + half) + 1))
        if rr.size and cc.size:
            sigma0[np.ix_(rr, cc)] *= strength
    return sigma0


def draw_blob(sigma0, centre_rc, radius_px, damping, irregular=True, rng=None):
    """Paint an irregular blob - algal bloom or a fixed source."""
    h, w = sigma0.shape
    r0, c0 = centre_rc
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(yy - r0, xx - c0)
    if irregular and rng is not None:
        angle = np.arctan2(yy - r0, xx - c0)
        wobble = 1.0 + 0.35 * np.sin(3 * angle + rng.uniform(0, 6.28)) \
                     + 0.2 * np.sin(7 * angle + rng.uniform(0, 6.28))
        mask = dist < radius_px * wobble
    else:
        mask = dist < radius_px
    sigma0[mask] *= damping
    return sigma0


def build_scene(
    height: int = 1400,
    width: int = 1400,
    sea_sigma0: float = 0.022,     # ~ -16.6 dB, typical VV open ocean
    enl: float = 4.4,
    seed: int = 7,
    include_lookalikes: bool = True,
):
    """Return (vv_linear, vh_linear, truth) for a synthetic scene."""
    rng = np.random.default_rng(seed)

    # Large-scale wind-field variation, so the sea is not uniformly flat.
    yy, xx = np.mgrid[0:height, 0:width] / max(height, width)
    background = sea_sigma0 * (1.0 + 0.18 * np.sin(4 * math.pi * xx) * np.cos(3 * math.pi * yy))
    vv = background.copy()

    truth: dict[str, dict] = {}

    # 1. The bilge dump: long, thin, tapering, ~8 dB damping.
    vv = draw_streak(vv, (430, 250), (505, 1150), width_px=26, damping=0.16)
    truth["bilge_dump"] = {
        "kind": "oil", "morphology": "linear",
        "row_range": [420, 515], "col_range": [250, 1150],
        "expected": "ACCEPT - real oil from a moving vessel",
    }

    if include_lookalikes:
        # 2. Algal bloom: round, irregular, weak damping (~3.5 dB).
        vv = draw_blob(vv, (980, 380), 95, damping=0.45, irregular=True, rng=rng)
        truth["algal_bloom"] = {
            "kind": "lookalike", "subtype": "biogenic_film",
            "centre_rc": [980, 380],
            "expected": "REJECT - round blob, weak damping",
        }

        # 3. Rain cell: near-circular, sharp edge, granular inside.
        vv = draw_blob(vv, (300, 1050), 70, damping=0.40, irregular=False)
        cell = np.zeros_like(vv, dtype=bool)
        yyc, xxc = np.mgrid[0:height, 0:width]
        cell[np.hypot(yyc - 300, xxc - 1050) < 70] = True
        vv[cell] *= rng.uniform(0.6, 1.5, size=cell.sum())  # granular texture
        truth["rain_cell"] = {
            "kind": "lookalike", "subtype": "rain_cell",
            "centre_rc": [300, 1050],
            "expected": "REJECT - circular with granular texture",
        }

        # 4. Internal waves: regular periodic banding.
        band_rows = slice(1150, 1330)
        band_cols = slice(600, 1250)
        cols = np.arange(band_cols.start, band_cols.stop)
        wave = 1.0 - 0.42 * (np.sin(cols / 11.0) > 0.35)
        vv[band_rows, band_cols] *= wave[None, :]
        truth["internal_waves"] = {
            "kind": "lookalike", "subtype": "internal_waves",
            "row_range": [1150, 1330], "col_range": [600, 1250],
            "expected": "REJECT - regular periodic banding",
        }

    # VH sits ~8 dB below VV over the sea; oil suppresses it harder still.
    vh = vv * 0.16
    oil_like = vv < (background * 0.35)
    vh[oil_like] *= 0.55

    vv = vv * gamma_speckle(vv.shape, enl, rng)
    vh = vh * gamma_speckle(vh.shape, enl, rng)
    return vv.astype(np.float32), vh.astype(np.float32), truth


def synth_ais(bbox, acquired_at, origin_lonlat, axis_deg=90.0, seed=11):
    """AIS for the demo: real voyages between real ports, passing our box.

    Deliberately long. A vessel seen crossing the scene was under way for
    hours beforehand and keeps going afterwards, and the map should show that
    whole passage rather than a stub around the slick.
    """
    from core.geo import bearing_deg, destination_point, haversine_km

    rng = np.random.default_rng(seed)
    release = acquired_at - timedelta(hours=12)
    rows = []

    def sail(mmsi, name, vtype, flag, start_pt, end_pt, arrive_at,
             speed_kn=12.0, step_min=20, gap_near=None, destination="",
             via=None, via_at=None):
        """Steam start -> (via) -> end, arriving at arrive_at.

        The via point exists because real passages are not great-circle
        straight lines: vessels route around traffic and weather. It is also
        what puts a track through the scene at all.
        """
        legs = [start_pt] + ([via] if via else []) + [end_pt]
        leg_km = speed_kn * 1.852 * step_min / 60.0

        # Build the whole path first so the arrival time lands exactly.
        path = []
        for a, b in zip(legs, legs[1:]):
            course = bearing_deg(a, b)
            steps = max(2, int(haversine_km(a, b) / leg_km))
            pt = a
            for _ in range(steps):
                path.append((pt, course))
                pt = destination_point(pt, course, leg_km)
        path.append((legs[-1], bearing_deg(legs[-2], legs[-1])))

        if via is not None and via_at is not None:
            # Anchor the timeline on the moment the vessel is at the via point,
            # not on its arrival. Which hour a ship passes the origin is the
            # whole temporality signal, so it has to be set deliberately
            # rather than falling out of the leg lengths.
            via_index = min(
                range(len(path)),
                key=lambda i: haversine_km(path[i][0], via),
            )
            depart = via_at - timedelta(minutes=step_min * via_index)
        else:
            depart = arrive_at - timedelta(minutes=step_min * (len(path) - 1))
        t = depart
        for pt, course in path:
            # A gap counts as "going dark" only if it happens near the origin.
            if gap_near is not None and haversine_km(pt, gap_near) < 15.0:
                t += timedelta(hours=3)
                gap_near = None
                continue
            rows.append({
                "MMSI": mmsi, "Name": name, "Ship type": vtype, "Flag": flag,
                "Destination": destination,
                "# Timestamp": t.strftime("%d/%m/%Y %H:%M:%S"),
                "Latitude": round(pt[1], 6), "Longitude": round(pt[0], 6),
                "SOG": round(speed_kn + rng.normal(0, 0.3), 1),
                "COG": round(course + rng.normal(0, 1.5), 1),
            })
            t += timedelta(minutes=step_min)

    # Ports either side of the scene, so each track is a real passage.
    colombo = (79.842, 6.951)
    kochi = (76.267, 9.966)
    mumbai = (72.842, 18.944)
    mangalore = (74.802, 12.922)
    male = (73.509, 4.175)
    salalah = (54.005, 16.940)

    # The dark runner: Colombo -> Mumbai, silent as it passes the origin.
    upstream = destination_point(origin_lonlat, (axis_deg + 180) % 360, 60)
    downstream = destination_point(origin_lonlat, axis_deg, 60)
    sail("419005678", "MV NIGHT PASSAGE", "Tanker", "PA", colombo, mumbai,
         acquired_at + timedelta(hours=20), speed_kn=11.5,
         gap_near=origin_lonlat, destination="MUMBAI",
         via=origin_lonlat, via_at=release)

    # A second vessel on a similar heading but offset - real traffic does not
    # ride identical tracks, and an exact overlap makes the ranking abstain
    # for the wrong reason.
    sail("419001234", "MV MERIDIAN STAR", "Cargo", "IN",
         destination_point(colombo, 320, 40), destination_point(mumbai, 200, 30),
         acquired_at + timedelta(hours=18), speed_kn=13.0,
         destination="NHAVA SHEVA",
         via=destination_point(origin_lonlat, (axis_deg + 90) % 360, 14),
         via_at=release + timedelta(minutes=40))

    # Crossing traffic: Kochi -> Male, roughly perpendicular to the slick.
    sail("419009012", "MV COASTAL TRADER", "Cargo", "IN", kochi, male,
         acquired_at + timedelta(hours=14), speed_kn=10.0, destination="MALE",
         via=destination_point(origin_lonlat, 200, 30),
         via_at=release + timedelta(hours=2))

    # Distant traffic, well outside the search radius.
    sail("419003456", "MV DISTANT HORIZON", "Fishing", "LK", mangalore, salalah,
         acquired_at + timedelta(hours=30), speed_kn=9.0, destination="SALALAH")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/demo_internal", help="output directory")
    ap.add_argument("--name", default="SYNTH_DEMO_001")
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-lookalikes", action="store_true")
    ap.add_argument(
        "--calm-wind", action="store_true",
        help="generate the low-wind look-alike scene: a dark patch a naive "
             "detector calls oil, which the wind check correctly rejects",
    )
    # Fully offshore in the Arabian Sea, west of Kerala. Deliberately clear of
    # the coastline: a demo bbox overlapping land has the land mask remove
    # part of the planted slick, which looks like a detector failure.
    ap.add_argument("--bbox", default="75.20,9.05,75.80,9.65",
                    help="min_lon,min_lat,max_lon,max_lat (default: Arabian Sea, offshore)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from ingest.raster import write_raster

    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox needs 4 comma-separated values")

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    vv, vh, truth = build_scene(
        height=args.size, width=args.size, seed=args.seed,
        include_lookalikes=not args.no_lookalikes,
        # A calm sea returns far less energy overall - that low backscatter is
        # exactly why oil becomes indistinguishable from water below ~3 m/s.
        sea_sigma0=0.006 if args.calm_wind else 0.022,
    )

    acquired_at = datetime(2025, 5, 25, 5, 42, tzinfo=timezone.utc)
    vv_path = write_raster(out_dir / f"{args.name}_vv.tif", vv, bbox)
    vh_path = write_raster(out_dir / f"{args.name}_vh.tif", vh, bbox)

    # Origin of the dump: the fresh (western) end of the streak.
    min_lon, min_lat, max_lon, max_lat = bbox
    origin_lon = min_lon + (250 / args.size) * (max_lon - min_lon)
    origin_lat = max_lat - (430 / args.size) * (max_lat - min_lat)

    ais_rows = synth_ais(bbox, acquired_at, (origin_lon, origin_lat))
    ais_path = out_dir / f"{args.name}_ais.csv"
    import csv

    with ais_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ais_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ais_rows)

    manifest = {
        "scene_id": args.name,
        "SYNTHETIC": True,
        "config": "configs/demo_calm_wind.yaml" if args.calm_wind else None,
        "scenario": "low-wind look-alike" if args.calm_wind else "bilge dump",
        "note": "Generated for pipeline testing. Never report numbers from this scene.",
        "acquired_at": acquired_at.isoformat(),
        "bbox": list(bbox),
        "vv_path": str(vv_path.relative_to(REPO_ROOT)),
        "vh_path": str(vh_path.relative_to(REPO_ROOT)),
        "ais_path": str(ais_path.relative_to(REPO_ROOT)),
        "orbit_direction": "DESCENDING",
        "truth": truth,
        "expected_origin": {"lon": round(origin_lon, 5), "lat": round(origin_lat, 5)},
    }
    manifest_path = out_dir / f"{args.name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote synthetic scene '{args.name}'")
    print(f"  VV      : {vv_path}  ({vv.shape[0]}x{vv.shape[1]})")
    print(f"  VH      : {vh_path}")
    print(f"  AIS     : {ais_path}  ({len(ais_rows)} positions, "
          f"{len({r['MMSI'] for r in ais_rows})} vessels)")
    print(f"  manifest: {manifest_path}")
    print(f"  features: {', '.join(truth)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
