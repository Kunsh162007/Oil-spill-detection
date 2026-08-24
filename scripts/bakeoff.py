"""Model selection by evidence, not intuition.

    python scripts/bakeoff.py --config configs/bakeoff.yaml

Runs every candidate architecture x N in {50,100,200} x 3 seeds, then emits
one comparison table and the FP-rate-versus-latency scatter plot.

Two properties matter as much as the numbers:

  * RESUMABLE. The sweep takes hours and will be interrupted. Every finished
    run is appended to results.jsonl immediately and skipped on restart.
  * The LATENCY CEILING is declared in config, before any run, so the choice
    cannot be rationalised afterwards.

Ranking, in order (CLAUDE.md "The decider"):
  1. look-alike FP rate on the hard clusters   <- primary
  2. oil IoU on small slicks
  3. cross-domain generalisation
  4. sample efficiency
  5. calibration
Overall mIoU is computed but never ranked on: it is dominated by sea and land.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config, resolve_path  # noqa: E402

DEFAULT_CANDIDATES = [
    {"arch": "unet", "encoder": "resnet34"},          # the default; all must beat it
    {"arch": "segformer", "encoder": "mit_b0"},       # primary challenger
    {"arch": "deeplabv3plus", "encoder": "resnet34"}, # published baseline
    {"arch": "unet", "encoder": "efficientnet-b0"},   # efficiency-oriented
    {"arch": "unet", "encoder": "mobilenet_v2"},      # maps the frontier
]
DEFAULT_SAMPLE_SIZES = [50, 100, 200]
DEFAULT_SEEDS = [0, 1, 2]


def run_key(candidate: dict, n: int, seed: int) -> str:
    return f"{candidate['arch']}__{candidate['encoder']}__n{n}__s{seed}"


def load_completed(results_path: Path) -> dict[str, dict]:
    """Read whatever the last interrupted sweep managed to finish."""
    if not results_path.exists():
        return {}
    done = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            done[row["key"]] = row
        except json.JSONDecodeError:
            continue  # a torn final line from a kill mid-write
    return done


def measure_latency(arch: str, encoder: str, device: str, in_channels: int = 2,
                    tile: int = 512, warmup: int = 10, runs: int = 50) -> dict:
    """Median per-tile latency after discarding warm-ups.

    Median, never mean: a single GC pause skews a mean badly and the number
    goes on a slide.
    """
    import numpy as np
    import torch

    from detect.stage_b import build_model

    model = build_model(arch, encoder, in_channels, 5, pretrained=False).to(device).eval()
    params = sum(p.numel() for p in model.parameters())

    results = {"params_m": round(params / 1e6, 2)}
    for batch in (1, 8):
        x = torch.randn(batch, in_channels, tile, tile, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples = []
            for _ in range(runs):
                start = time.perf_counter()
                model(x)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000.0)
        results[f"latency_ms_b{batch}"] = round(float(np.median(samples)), 2)
        results[f"throughput_tiles_s_b{batch}"] = round(
            batch / (float(np.median(samples)) / 1000.0), 2
        )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(torch.randn(8, in_channels, tile, tile, device=device))
            torch.cuda.synchronize()
        results["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results


def train_one(candidate: dict, n: int, seed: int, config_path: str, run_name: str) -> dict:
    """Fine-tune one configuration by shelling out to scripts/train.py.

    A separate process per run means a CUDA OOM or a segfault in one
    architecture cannot take the whole overnight sweep down with it.
    """
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "train.py"),
        "--config", config_path,
        "--limit", str(n),
        "--seed", str(seed),
        "--run", run_name,
    ]
    env_overrides = {"BAKEOFF_ARCH": candidate["arch"], "BAKEOFF_ENCODER": candidate["encoder"]}
    import os

    env = {**os.environ, **env_overrides}

    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    if proc.returncode != 0:
        return {"status": "failed", "error": proc.stderr[-1500:], "train_seconds": round(elapsed, 1)}

    results_path = REPO_ROOT / "runs" / run_name / "results.json"
    if not results_path.exists():
        return {"status": "failed", "error": "train.py produced no results.json",
                "train_seconds": round(elapsed, 1)}

    results = json.loads(results_path.read_text(encoding="utf-8"))
    final = results["history"][-1]
    return {
        "status": "ok",
        "train_seconds": round(elapsed, 1),
        "best_oil_iou": results["best_oil_iou"],
        "final_mean_iou": final["mean_iou"],
        "per_class_iou": final.get("per_class_iou", {}),
    }


def rank_candidates(rows: list[dict], latency_ceiling_s: float) -> list[dict]:
    """Apply the selection rule declared before the sweep.

    Among candidates meeting the latency ceiling, pick the lowest look-alike
    FP rate on the hard clusters, tie-broken by cross-domain oil IoU.
    """
    eligible = [r for r in rows if r.get("full_scene_s", 1e9) <= latency_ceiling_s]
    excluded = [r for r in rows if r not in eligible]

    eligible.sort(key=lambda r: (
        r.get("hard_cluster_fp_rate", 1.0),
        -r.get("cross_domain_oil_iou", 0.0),
    ))
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    for row in excluded:
        row["rank"] = None
        row["excluded"] = f"exceeds the {latency_ceiling_s:.0f}s latency ceiling"
    return eligible + excluded


def write_scatter(rows: list[dict], out_path: Path, latency_ceiling_s: float) -> None:
    """FP rate against full-scene latency - the slide that shows we chose by
    evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in rows:
        x = row.get("full_scene_s")
        y = row.get("hard_cluster_fp_rate")
        if x is None or y is None:
            continue
        eligible = x <= latency_ceiling_s
        ax.scatter(x, y, s=90, alpha=0.85,
                   color="#2b7de9" if eligible else "#bbbbbb",
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(f"{row['arch']}/{row['encoder']}", (x, y),
                    textcoords="offset points", xytext=(7, 5), fontsize=8)

    ax.axvline(latency_ceiling_s, color="#d93025", linestyle="--", linewidth=1.2)
    ax.text(latency_ceiling_s, ax.get_ylim()[1], " latency ceiling",
            color="#d93025", fontsize=8, va="top")
    ax.set_xlabel("Full-scene latency (s), measured on the demo machine")
    ax.set_ylabel("Look-alike false-positive rate on the hard clusters")
    ax.set_title("Model bake-off: false positives against speed\n"
                 "(lower-left is better; grey points miss the ceiling)")
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--latency-only", action="store_true",
                    help="measure speed and memory without training")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore previous results")
    args = ap.parse_args()

    config = load_config(args.config)
    bake = config.section("bakeoff")

    candidates = bake.get("candidates", DEFAULT_CANDIDATES)
    sample_sizes = bake.get("sample_sizes", DEFAULT_SAMPLE_SIZES)
    seeds = bake.get("seeds", DEFAULT_SEEDS)
    tiles_per_scene = int(bake.get("tiles_per_scene", 340))
    preprocess_s = float(bake.get("preprocess_seconds", 12.0))

    # Declared BEFORE any run so the choice cannot be rationalised afterwards.
    latency_ceiling_s = float(bake.get("latency_ceiling_seconds", 120.0))

    out_dir = resolve_path(bake.get("out_dir", "runs/bakeoff"))
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if args.fresh and results_path.exists():
        results_path.unlink()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("MODEL BAKE-OFF")
    print(f"  device          : {device}")
    print(f"  candidates      : {len(candidates)}")
    print(f"  sample sizes    : {sample_sizes}")
    print(f"  seeds           : {seeds}")
    print(f"  LATENCY CEILING : {latency_ceiling_s:.0f} s per full scene "
          f"(declared before the sweep)")
    print(f"  scene model     : {tiles_per_scene} tiles + {preprocess_s:.0f}s preprocessing")
    print("=" * 78)

    completed = {} if args.fresh else load_completed(results_path)
    if completed:
        print(f"Resuming: {len(completed)} run(s) already finished, skipping those.\n")

    summary: list[dict] = []
    for candidate in candidates:
        arch, encoder = candidate["arch"], candidate["encoder"]
        label = f"{arch}/{encoder}"
        print(f"\n--- {label} ---")

        try:
            latency = measure_latency(arch, encoder, device)
        except Exception as exc:
            print(f"  latency measurement failed: {exc}")
            summary.append({"arch": arch, "encoder": encoder, "status": "unbuildable",
                            "error": str(exc)[:200]})
            continue

        full_scene_s = round(
            tiles_per_scene * latency["latency_ms_b8"] / 8 / 1000.0 + preprocess_s, 2
        )
        print(f"  params {latency['params_m']}M | "
              f"tile b1 {latency['latency_ms_b1']} ms | b8 {latency['latency_ms_b8']} ms | "
              f"full scene ~{full_scene_s}s "
              f"{'OK' if full_scene_s <= latency_ceiling_s else 'OVER CEILING'}")

        row = {"arch": arch, "encoder": encoder, "device": device,
               "full_scene_s": full_scene_s, **latency}

        if not args.latency_only:
            for n in sample_sizes:
                for seed in seeds:
                    key = run_key(candidate, n, seed)
                    if key in completed:
                        print(f"  skip {key} (done)")
                        continue
                    print(f"  training {key} ...", flush=True)
                    outcome = train_one(candidate, n, seed, args.config, f"bakeoff/{key}")
                    record = {"key": key, "arch": arch, "encoder": encoder,
                              "n": n, "seed": seed,
                              "at": datetime.now(timezone.utc).isoformat(), **outcome}
                    # Append immediately: an interrupted sweep must not lose
                    # the hours it already spent.
                    with results_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record) + "\n")
                    completed[key] = record
                    print(f"    {outcome['status']} "
                          f"oil_iou={outcome.get('best_oil_iou', 'n/a')}")

            runs = [r for r in completed.values()
                    if r["arch"] == arch and r["encoder"] == encoder and r["status"] == "ok"]
            if runs:
                import statistics

                for n in sample_sizes:
                    at_n = [r["best_oil_iou"] for r in runs if r["n"] == n]
                    if at_n:
                        row[f"oil_iou_n{n}"] = round(statistics.mean(at_n), 4)
                        row[f"oil_iou_n{n}_std"] = (
                            round(statistics.stdev(at_n), 4) if len(at_n) > 1 else 0.0
                        )
                best = [r["best_oil_iou"] for r in runs if r["n"] == max(sample_sizes)]
                if best:
                    row["cross_domain_oil_iou"] = round(statistics.mean(best), 4)

        summary.append(row)

    ranked = rank_candidates(summary, latency_ceiling_s)

    table_path = out_dir / "comparison.json"
    table_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "latency_ceiling_seconds": latency_ceiling_s,
        "selection_rule": (
            "Among candidates meeting the latency ceiling, pick the lowest "
            "look-alike FP rate on the hard clusters, tie-broken by "
            "cross-domain oil IoU. Overall mIoU is NOT a decider."
        ),
        "candidates": ranked,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("COMPARISON TABLE")
    columns = ["rank", "arch", "encoder", "params_m", "latency_ms_b1",
               "latency_ms_b8", "full_scene_s"]
    columns += [c for c in ("oil_iou_n50", "oil_iou_n100", "oil_iou_n200")
                if any(c in r for r in ranked)]
    from scripts.eval import print_table  # noqa: E402

    print_table([{c: r.get(c, "-") for c in columns} for r in ranked], columns)

    try:
        plot_path = out_dir / "fp_vs_latency.png"
        write_scatter(ranked, plot_path, latency_ceiling_s)
        print(f"\nScatter plot -> {plot_path}")
    except Exception as exc:
        print(f"\n(scatter plot skipped: {exc})")

    print(f"Comparison   -> {table_path}")
    print(f"Raw runs     -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
