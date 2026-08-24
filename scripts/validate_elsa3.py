"""Validate the pipeline against the MSC ELSA 3 incident.

    python scripts/validate_elsa3.py

MSC ELSA 3 sank on 25 May 2025 off Kochi carrying furnace oil and diesel; the
wreck leaked for weeks. The vessel and the position are public record, so this
is the one case where our answer can be checked against documented truth
rather than against another model.

What is being tested is NOT "did we name the ship". The ship is known and it
sank - there is no attribution puzzle. What must hold is:

  1. we detect a slick at all on the real scene
  2. an independent incident registry corroborates it
  3. BACKWARD DRIFT moves the estimate toward the wreck
  4. no passing vessel is accused for what is a documented wreck

Point 3 matters more than raw distance, and the two are not the same thing.
This scene is 3 days after the sinking; oil drifting at even 0.3 m/s covers
~70 km in that time, so a slick tens of km from the wreck is what the physics
predicts, not a miss. Scoring on distance alone would mark correct behaviour
as failure. What we can legitimately ask is whether rewinding the drift moves
the estimate toward the wreck rather than away from it.
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Documented wreck position: 09 18.75'N 076 08.16'E
WRECK_LON, WRECK_LAT = 76.136, 9.3125
WRECK_NAME = "MSC ELSA 3"
SINKING_DATE = datetime(2025, 5, 25)


def main() -> int:
    from core.config import load_config
    from core.contracts import Scene
    from core.geo import haversine_km
    from decision.pipeline import analyse_scene
    from detect.incidents import load_registry
    from detect.wind import build_wind_lookup

    manifests = sorted(glob.glob(str(REPO_ROOT / "data/demo_finale/*.json")))
    if not manifests:
        print(
            "No scenes in data/demo_finale. Fetch them first:\n"
            "  python scripts/fetch_aws.py --bbox 75.7,8.9,76.6,9.8 "
            "--date 2025-05-27 --days 4 --out-dir data/demo_finale "
            "--config configs/elsa3.yaml",
            file=sys.stderr,
        )
        return 1

    config = load_config("configs/elsa3.yaml")
    registry = load_registry(REPO_ROOT / "data/reference/noaa_incidents.csv")
    setattr(config, "_incident_registry", registry)
    wind = build_wind_lookup(config)

    print("=" * 74)
    print(f"VALIDATION: {WRECK_NAME}, Kerala, {SINKING_DATE:%d %B %Y}")
    print(f"Documented wreck position: {WRECK_LAT:.4f}N {WRECK_LON:.4f}E")
    print("=" * 74)

    checks = {
        "detected": False,
        "corroborated": False,
        "drift_points_to_wreck": False,
        "no_false_accusation": True,
    }
    best_distance = float("inf")
    best_origin_distance = float("inf")
    drift_pairs: list[tuple[float, float]] = []

    for path in manifests:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        scene = Scene(
            scene_id=data["scene_id"],
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            bbox=tuple(data["bbox"]),
            vv_path=REPO_ROOT / data["vv_path"],
            vh_path=REPO_ROOT / data["vh_path"] if data.get("vh_path") else None,
            orbit_direction=data.get("orbit_direction", "UNKNOWN"),
        )
        result = analyse_scene(scene, config, wind_lookup=wind, ais_tracks=None)
        days_after = (scene.acquired_at.date() - SINKING_DATE.date()).days

        print(f"\nScene {scene.scene_id[:52]}")
        print(f"  acquired {scene.acquired_at:%Y-%m-%d %H:%M} UTC "
              f"({days_after} days after the sinking)")
        print(f"  bbox {[round(v, 3) for v in scene.bbox]}  "
              f"dual-pol={scene.is_dual_pol}")
        print(f"  regions={result.stats['n_regions']}  "
              f"confirmed={result.stats['n_confirmed']}  "
              f"rejected={result.stats['n_rejected']}")

        # Report the largest few; a full scene produces many small candidates.
        shown = sorted(result.confirmed, key=lambda c: c.area_km2, reverse=True)[:6]

        for candidate in result.confirmed:
            checks["detected"] = True
            distance = haversine_km(candidate.centroid, (WRECK_LON, WRECK_LAT))
            best_distance = min(best_distance, distance)

            attribution = next(
                a for a in result.attributions
                if a.candidate_id == candidate.candidate_id
            )
            corroboration = attribution.evidence.get("corroboration", {})
            confidence = attribution.evidence.get("confidence", {})

            if corroboration.get("confirmed"):
                checks["corroborated"] = True

            # Naming a vessel here would be the failure mode: the source is a
            # documented wreck, not a passing ship.
            if not attribution.abstained and attribution.candidates:
                checks["no_false_accusation"] = False

            origin_distance = None
            if attribution.origin is not None:
                origin_distance = haversine_km(
                    (attribution.origin.lon, attribution.origin.lat),
                    (WRECK_LON, WRECK_LAT),
                )
                best_origin_distance = min(best_origin_distance, origin_distance)
                drift_pairs.append((distance, origin_distance))
                if origin_distance < distance:
                    checks["drift_points_to_wreck"] = True

            if candidate in shown:
                arrow = ""
                if origin_distance is not None:
                    direction = "closer" if origin_distance < distance else "further"
                    arrow = f"  ->  origin {origin_distance:6.1f} km ({direction})"
                print(f"    {candidate.candidate_id[-12:]}  "
                      f"{candidate.area_km2:6.2f} km2  "
                      f"P(oil) {candidate.p_oil:.2f}  "
                      f"{distance:6.1f} km from wreck  "
                      f"[{confidence.get('tier', '?')}]{arrow}")
                if corroboration.get("confirmed"):
                    print(f"        registry: "
                          f"{corroboration['matches'][0]['reason'][:74]}")

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    labels = {
        "detected": "a slick was detected on the real Sentinel-1 scene",
        "corroborated": "independently corroborated by the incident registry",
        "drift_points_to_wreck": "backward drift moves the estimate toward the wreck",
        "no_false_accusation": "no vessel was accused for a documented wreck",
    }
    for key, label in labels.items():
        print(f"  [{'PASS' if checks[key] else 'FAIL'}] {label}")

    if best_distance < float("inf"):
        print(f"\n  closest detection to the wreck : {best_distance:6.1f} km")
    if best_origin_distance < float("inf"):
        print(f"  closest drift origin to wreck  : {best_origin_distance:6.1f} km")
    if drift_pairs:
        closer = sum(1 for observed, origin in drift_pairs if origin < observed)
        share = closer / len(drift_pairs)
        print(f"  drift moved {closer}/{len(drift_pairs)} estimates toward the wreck "
              f"({share:.0%})")
        currents = str(config.get("drift.currents_source", "")).lower()
        if currents == "synthetic":
            print(
                "\n  CAVEAT: currents are SYNTHETIC, so the drift direction is close"
                "\n  to arbitrary and this check carries little weight - roughly half"
                "\n  the estimates would move toward the wreck by chance alone. Supply"
                "\n  real CMEMS currents before quoting this as evidence."
            )

    print(
        "\n  Note: MSC ELSA 3 is a known WRECK. The correct behaviour is to detect"
        "\n  the oil and decline to blame a passing ship - a system that named a"
        "\n  vessel here would be failing, not succeeding."
        "\n"
        "\n  Distance from the wreck is NOT the metric. This scene is 3 days after"
        "\n  the sinking, and oil drifting at 0.3 m/s covers ~70 km in that time,"
        "\n  so a slick tens of km away is what the physics predicts."
    )
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
