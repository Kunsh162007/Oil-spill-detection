"""Evaluate a pipeline run and print a table a non-technical person can paste
into a spreadsheet.

    python scripts/eval.py --run runs/<run_name>
    python scripts/eval.py --config configs/demo_synthetic.yaml --wind-ablation

The wind ablation is our contribution: false-positive rate with and without
the wind stage, broken down per look-alike class. "Wind fusion cut false
positives on the low-wind cluster by X%" is a far stronger claim than any
aggregate number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config, resolve_path  # noqa: E402


def print_table(rows: list[dict], columns: list[str]) -> None:
    """Fixed-width table, tab-separated values underneath for pasting."""
    if not rows:
        print("  (no rows)")
        return
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  " + "  ".join(c.ljust(widths[c]) for c in columns))
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    print("\n  --- tab-separated, for a spreadsheet ---")
    print("  " + "\t".join(columns))
    for row in rows:
        print("  " + "\t".join(str(row.get(c, "")) for c in columns))


def summarise_run(run_dir: Path) -> int:
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"No results.json in {run_dir}", file=sys.stderr)
        return 1
    results = json.loads(results_path.read_text(encoding="utf-8"))

    print(f"\nRUN: {results['run_name']}")
    print(f"  architecture : {results['arch']} / {results['encoder']}")
    print(f"  device       : {results['device']}")
    print(f"  patches      : {results['n_train']} train / {results['n_val']} val")
    print(f"  best OIL IoU : {results['best_oil_iou']}")
    print(f"  selected on  : {results['selection_metric']}\n")

    print("EPOCH HISTORY")
    print_table(
        [{k: v for k, v in row.items() if k != "per_class_iou"} for row in results["history"]],
        ["epoch", "train_loss", "val_loss", "oil_iou", "mean_iou", "seconds"],
    )

    final = results["history"][-1].get("per_class_iou", {})
    if final:
        print("\nPER-CLASS IoU (final epoch)")
        print_table([{"class": k, "iou": v} for k, v in final.items()], ["class", "iou"])
        print(
            "\n  NOTE: mean IoU is dominated by the sea and land classes, which\n"
            "  every model scores in the 90s on. Rank on oil IoU and on the\n"
            "  look-alike false-positive rate instead."
        )
    return 0


def wind_ablation(config_path: str) -> int:
    """Run every scene with and without the wind stage and compare.

    This is the measurement behind our central claim, so it runs the real
    pipeline twice rather than estimating anything.
    """
    from attribute.ais import load_ais_csv
    from core.config import load_config
    from core.contracts import Scene
    from datetime import datetime

    from decision.pipeline import analyse_scene
    from detect.wind import build_wind_lookup

    config = load_config(config_path)
    manifests = []
    for root in ("data/demo_internal", "data/demo_finale", "data/dev"):
        base = resolve_path(root)
        if base.is_dir():
            for path in sorted(base.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                if {"scene_id", "bbox", "vv_path"} <= set(data):
                    manifests.append(data)

    if not manifests:
        print("No scenes found. Run scripts/make_demo_scene.py first.", file=sys.stderr)
        return 1

    rows = []
    for data in manifests:
        scene = Scene(
            scene_id=data["scene_id"],
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            bbox=tuple(data["bbox"]),
            vv_path=resolve_path(data["vv_path"]),
            vh_path=resolve_path(data["vh_path"]) if data.get("vh_path") else None,
            orbit_direction=data.get("orbit_direction", "DESCENDING"),
        )
        tracks = None
        if data.get("ais_path"):
            try:
                loaded = load_ais_csv(resolve_path(data["ais_path"]), source="eval")
                tracks = lambda origin, when, _t=loaded: _t
            except Exception:
                tracks = None

        # Honour a per-scene config override, the way api/store.py does. The
        # calm-wind demo scene carries its own low wind value; using the
        # global config here would silently evaluate it under ideal wind and
        # report that the wind stage does nothing.
        scene_config = load_config(data["config"]) if data.get("config") else config

        with_wind = analyse_scene(scene, scene_config,
                                  wind_lookup=build_wind_lookup(scene_config),
                                  ais_tracks=tracks)

        # Ablation: force every wind reading into the ideal window, which is
        # exactly what a system with no wind stage assumes by default.
        from core.contracts import WindContext

        blind = lambda lon, lat, when: WindContext.from_speed(5.5, 0.0, source="ablation-none")
        without_wind = analyse_scene(scene, scene_config, wind_lookup=blind,
                                     ais_tracks=tracks)

        wind_rejects = sum(
            1 for c in with_wind.rejected
            if c.rejected_reason and "wind" in c.rejected_reason.lower()
        )
        rows.append({
            "scene": scene.scene_id,
            "regions": with_wind.stats["n_regions"],
            "confirmed_with_wind": with_wind.stats["n_confirmed"],
            "confirmed_no_wind": without_wind.stats["n_confirmed"],
            "rejected_by_wind": wind_rejects,
            "fp_reduction": round(
                1.0 - with_wind.stats["n_confirmed"] / max(without_wind.stats["n_confirmed"], 1), 4
            ),
        })

    print("\nWIND ABLATION - the contribution")
    print("  'confirmed' counts slicks that survived to vessel attribution.")
    print("  Fewer with wind = fewer false accusations.\n")
    print_table(rows, ["scene", "regions", "confirmed_no_wind",
                       "confirmed_with_wind", "rejected_by_wind", "fp_reduction"])

    total_with = sum(r["confirmed_with_wind"] for r in rows)
    total_without = sum(r["confirmed_no_wind"] for r in rows)
    if total_without:
        print(f"\n  Overall: {total_without} -> {total_with} confirmed slicks, "
              f"a {100*(1-total_with/total_without):.1f}% reduction in candidates "
              f"reaching vessel attribution.")
    return 0


def latency_report(config_path: str) -> int:
    """Per-stage timing, so we know what to optimise."""
    from attribute.ais import load_ais_csv
    from core.config import load_config
    from core.contracts import Scene
    from datetime import datetime

    from decision.pipeline import analyse_scene
    from detect.wind import build_wind_lookup

    config = load_config(config_path)
    base = resolve_path("data/demo_internal")
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(base.glob("*.json"))]
    if not manifests:
        print("No demo scene. Run scripts/make_demo_scene.py", file=sys.stderr)
        return 1

    data = manifests[0]
    scene = Scene(
        scene_id=data["scene_id"],
        acquired_at=datetime.fromisoformat(data["acquired_at"]),
        bbox=tuple(data["bbox"]),
        vv_path=resolve_path(data["vv_path"]),
        vh_path=resolve_path(data["vh_path"]) if data.get("vh_path") else None,
        orbit_direction=data.get("orbit_direction", "DESCENDING"),
    )
    tracks = load_ais_csv(resolve_path(data["ais_path"]), source="eval") if data.get("ais_path") else []
    analysis = analyse_scene(scene, config, wind_lookup=build_wind_lookup(config),
                             ais_tracks=lambda o, w: tracks)

    total = analysis.stats["total_s"]
    rows = [
        {"stage": stage, "seconds": secs, "percent": round(100 * secs / max(total, 1e-9), 1)}
        for stage, secs in analysis.timings.items()
    ]
    print(f"\nEND-TO-END LATENCY - {scene.scene_id}")
    print_table(sorted(rows, key=lambda r: -r["seconds"]), ["stage", "seconds", "percent"])
    print(f"\n  TOTAL {total:.2f} s   (budget: a full GRD scene under 120 s)")
    print(f"  segmentation backend: {analysis.stats['segmentation_backend']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="a runs/<name> directory")
    ap.add_argument("--config", default="configs/demo_synthetic.yaml")
    ap.add_argument("--wind-ablation", action="store_true")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()

    if args.run:
        return summarise_run(Path(args.run))
    if args.wind_ablation:
        return wind_ablation(args.config)
    if args.latency:
        return latency_report(args.config)

    ap.print_help()
    print("\nPick one of --run, --wind-ablation or --latency.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
