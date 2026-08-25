"""Analyse every scene at build time and cache the result.

A 512 MB container cannot run the pipeline on demand: rasterio arrays plus
speckle filtering exceed the headroom and the worker is OOM-killed mid-request,
which surfaces as a 502 with no body and no log line.

The analysis RESULT, though, is tiny - polygons, scores, vessel tracks, a few
hundred kilobytes per scene, with no arrays at all. So the work happens once
during `docker build`, where memory is plentiful, and the container only ever
deserialises the answer.

What this costs: the container can no longer analyse a NEW scene fetched at
runtime. That needs the memory this design is avoiding, so on a free tier it
is not available either way.

    python scripts/precompute.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIRNAME = "precomputed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="override the config; defaults to each scene's own")
    ap.add_argument("--out-dir", default=f"data/{CACHE_DIRNAME}")
    args = ap.parse_args()

    from core.config import load_config
    from core.contracts import Scene
    from decision.pipeline import analyse_scene
    from detect.incidents import load_registry
    from detect.wind import build_wind_lookup

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # The store discovers manifests in these roots; mirror that here so a scene
    # visible to the API is a scene that gets precomputed.
    roots = ["data/live", "data/demo_internal", "data/demo_finale",
             "data/dev", "data/dev/scenes"]
    manifests: list[dict] = []
    for root in roots:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if {"scene_id", "bbox", "vv_path"} <= set(data):
                manifests.append(data)

    if not manifests:
        print("No scene manifests found - nothing to precompute.", file=sys.stderr)
        return 0

    registry = None
    registry_path = REPO_ROOT / "data" / "reference" / "noaa_incidents.csv"
    try:
        registry = load_registry(registry_path if registry_path.exists() else None)
    except Exception as exc:
        print(f"WARNING: incident registry unavailable ({exc}); "
              f"detections will not be corroborated")

    ok, failed = 0, 0
    for data in manifests:
        scene_id = data["scene_id"]
        config = load_config(args.config or data.get("config") or None)
        if registry is not None:
            setattr(config, "_incident_registry", registry)

        scene = Scene(
            scene_id=scene_id,
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            bbox=tuple(data["bbox"]),
            vv_path=REPO_ROOT / data["vv_path"],
            vh_path=REPO_ROOT / data["vh_path"] if data.get("vh_path") else None,
            orbit_direction=data.get("orbit_direction", "DESCENDING"),
        )

        ais_tracks = None
        if data.get("ais_path"):
            from attribute.ais import load_ais_csv
            try:
                tracks = load_ais_csv(REPO_ROOT / data["ais_path"], source="precompute")
                ais_tracks = lambda origin, when, _t=tracks: _t
            except Exception as exc:
                print(f"  {scene_id}: AIS unavailable ({exc})")

        started = time.perf_counter()
        try:
            analysis = analyse_scene(
                scene, config,
                wind_lookup=build_wind_lookup(config),
                ais_tracks=ais_tracks,
            )
        except Exception as exc:
            print(f"  FAILED {scene_id}: {type(exc).__name__}: {exc}")
            failed += 1
            continue

        # The tile grid holds the full pixel arrays and is only needed during
        # analysis. Dropping it turns a multi-megabyte pickle into a small one,
        # which is the whole point of caching the result rather than the input.
        analysis.stats.pop("ingest", None)

        target = out_dir / f"{scene_id}.pkl"
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(analysis, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(target)

        size_kb = target.stat().st_size / 1024
        print(f"  {scene_id}: {analysis.stats.get('n_confirmed', 0)} confirmed, "
              f"{analysis.stats.get('n_rejected', 0)} rejected "
              f"({time.perf_counter() - started:.1f}s, {size_kb:.0f} KB)")
        ok += 1

    total_kb = sum(p.stat().st_size for p in out_dir.glob("*.pkl")) / 1024
    print(f"\nPrecomputed {ok} scene(s), {failed} failed, {total_kb:.0f} KB total")
    print(f"-> {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
