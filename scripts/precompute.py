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
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIRNAME = "precomputed"


def _portable(analysis):
    """A copy of the analysis whose paths survive a move between machines.

    pathlib pickles the CONCRETE class, so a Path built on Windows arrives as
    pathlib.WindowsPath and Linux refuses to instantiate it - the cache is
    written on a developer's machine and read inside a Linux container, so
    every entry fails there with NotImplementedError. Nothing in the API opens
    these paths; they exist for provenance. Storing them relative to the repo
    as PurePosixPath keeps them readable and correct on both.
    """
    import dataclasses

    def portable(value):
        if value is None:
            return None
        text = str(value)
        try:
            text = str(Path(value).resolve().relative_to(REPO_ROOT))
        except (ValueError, OSError):
            pass
        return PurePosixPath(text.replace("\\", "/"))

    scene = dataclasses.replace(
        analysis.scene,
        vv_path=portable(analysis.scene.vv_path),
        vh_path=portable(analysis.scene.vh_path),
    )
    return dataclasses.replace(analysis, scene=scene)


def _normalise_existing(out_dir: Path) -> int:
    """Rewrite already-cached analyses with portable paths."""
    fixed = 0
    for path in sorted(out_dir.glob("*.pkl")):
        try:
            with path.open("rb") as fh:
                analysis = pickle.load(fh)
        except Exception as exc:
            print(f"  {path.name}: unreadable ({exc})")
            continue
        if isinstance(analysis.scene.vv_path, PurePosixPath):
            continue
        with path.open("wb") as fh:
            pickle.dump(_portable(analysis), fh, protocol=pickle.HIGHEST_PROTOCOL)
        fixed += 1
    print(f"  normalised {fixed} cached analysis/analyses to portable paths")
    return fixed


def _write_world_index(out_dir: Path) -> None:
    """Build the map's world index once, so the API never has to.

    Serving /api/slicks means holding EVERY scene's analysis in memory at the
    same time. That is what exhausts a small container - not the pipeline, the
    results - and it happens on the map's very first request. Writing the
    finished GeoJSON here turns that into a single file read. Age fields are
    recomputed per request by api.serialize.refresh_ages, so a cached index
    never claims a stale detection is current.
    """
    from api.serialize import world_index

    analyses = []
    for path in sorted(out_dir.glob("*.pkl")):
        try:
            with path.open("rb") as fh:
                analyses.append(pickle.load(fh))
        except Exception as exc:
            print(f"  WARNING: {path.name} unreadable for the index ({exc})")

    if not analyses:
        print("  WARNING: no analyses to index; the API will build it live")
        return

    payload = world_index(analyses)
    target = out_dir / "world_index.json"
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    tmp.replace(target)
    print(f"  world index: {payload['meta']['n_slicks']} slick(s) across "
          f"{payload['meta']['n_scenes']} scene(s), "
          f"{target.stat().st_size / 1024:.0f} KB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="override the config; defaults to each scene's own")
    ap.add_argument("--out-dir", default=f"data/{CACHE_DIRNAME}")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave scenes that already have a cached result alone. "
                         "The container image ships analyses for scenes whose "
                         "imagery it does not carry, so re-analysing them there "
                         "would fail and destroy a good cache entry.")
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

    ok, failed, skipped = 0, 0, 0
    for data in manifests:
        scene_id = data["scene_id"]
        if args.skip_existing and (out_dir / f"{scene_id}.pkl").exists():
            skipped += 1
            continue
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
            pickle.dump(_portable(analysis), fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(target)

        size_kb = target.stat().st_size / 1024
        print(f"  {scene_id}: {analysis.stats.get('n_confirmed', 0)} confirmed, "
              f"{analysis.stats.get('n_rejected', 0)} rejected "
              f"({time.perf_counter() - started:.1f}s, {size_kb:.0f} KB)")
        ok += 1

    _normalise_existing(out_dir)
    _write_world_index(out_dir)

    total_kb = sum(p.stat().st_size for p in out_dir.glob("*.pkl")) / 1024
    cached = len(list(out_dir.glob("*.pkl")))
    print("")
    print(f"Precomputed {ok} scene(s), {skipped} already cached, {failed} failed")
    print(f"{cached} scene(s) in cache, {total_kb:.0f} KB total")
    print(f"-> {out_dir}")
    # A run that only skipped still leaves a usable cache, so a build
    # that re-analyses nothing is still a success.
    return 0 if cached else 1


if __name__ == "__main__":
    raise SystemExit(main())
