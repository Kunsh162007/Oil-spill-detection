"""Render the current architecture as a top-to-bottom JPEG flowchart.

    python scripts/make_diagram.py --out docs/architecture.jpg

Drawn from the code as it stands, not from the original design. Where a stage
is not what the plan called for - the classical detector standing in for the
trained U-Net, hand-set priors standing in for a fitted look-alike model - the
box says so. A diagram of the intended system would be worse than none.

Two layout lessons are baked in. Boxes size themselves to their text, because
hard-coded heights spilled the last line through the border. And the flow reads
straight down a single spine rather than across four columns, which was compact
and unreadable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

INK = "#111820"
MUTED = "#5b6b7a"
LINE = "#b9c4cf"
SOURCE = "#1f6f8b"      # external data
STAGE = "#26413c"       # pipeline
PHYSICS = "#8a4b1e"     # the differentiator
SERVE = "#3b3663"       # serving
WARN = "#9a2f2f"        # limits, and what is not as designed

LINE_H = 1.38           # vertical space one body line needs
HEAD_H = 3.1            # title plus the gap under it
PAD_H = 1.5
GAP = 3.8               # generous: the flow has to read at a glance


def draw(out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.5, 31.5), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(-72, 172)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf7")

    def height(body):
        return HEAD_H + ((body.count("\n") + 1) if body else 0) * LINE_H + PAD_H

    def box(x, top, w, title, body="", colour=STAGE, fs=7.5, tfs=9.4, align="center"):
        h = height(body)
        bottom = top - h
        for fill, edge, z in ((colour, "none", 2), ("none", colour, 3)):
            ax.add_patch(mp.FancyBboxPatch(
                (x, bottom), w, h, boxstyle="round,pad=0.35,rounding_size=0.9",
                linewidth=1.3, edgecolor=edge, facecolor=fill,
                alpha=.05 if z == 2 else 1.0, zorder=z))
        tx = x + w / 2 if align == "center" else x + 1.8
        ha = "center" if align == "center" else "left"
        ax.text(tx, top - 1.2, title, ha=ha, va="top", fontsize=tfs,
                color=colour, fontweight="bold", zorder=4)
        if body:
            ax.text(tx, top - HEAD_H, body, ha=ha, va="top", fontsize=fs,
                    color=INK, linespacing=1.55, zorder=4)
        return bottom

    def stack(x, top, w, items, colour, fs=7.5, tfs=9.4, gap=GAP,
              arrows=False, align="center"):
        spans, cursor = [], top
        for title, body in items:
            bottom = box(x, cursor, w, title, body, colour, fs, tfs, align)
            spans.append((cursor, bottom))
            cursor = bottom - gap
        if arrows:
            for (_, bot), (nxt, _) in zip(spans, spans[1:]):
                ax.annotate("", xy=(x + w / 2, nxt), xytext=(x + w / 2, bot),
                            zorder=1, arrowprops=dict(arrowstyle="-|>", color=LINE,
                                                      linewidth=1.9))
        return spans

    def down(x, y_from, y_to, colour, label=""):
        ax.annotate("", xy=(x, y_to), xytext=(x, y_from), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=2.2))
        if label:
            ax.text(x + 1.6, (y_from + y_to) / 2, label, fontsize=7.8, color=MUTED,
                    style="italic", va="center", ha="left")

    def band(y, label, colour, x=3):
        ax.text(x, y, label, fontsize=9.8, color=colour, fontweight="bold", va="top")

    # ---- header ----------------------------------------------------------
    ax.text(3, 169.6, "Oil Spill Detection & Vessel Attribution",
            fontsize=20, color=INK, fontweight="bold", va="top")
    ax.text(3, 165.3, "Architecture as deployed   ·   every input real   ·   "
                      "oilspill.onrender.com", fontsize=10.4, color=MUTED, va="top")
    ax.plot([3, 97], [163.0, 163.0], color=LINE, linewidth=1.2)

    COL_X, COL_W = 3.0, 60.0
    SPINE = COL_X + COL_W / 2
    SIDE_X, SIDE_W = 67.0, 30.0

    # ---- 1. data in ------------------------------------------------------
    band(160.4, "1 · DATA IN      fetched, never invented", SOURCE)
    src = stack(COL_X, 156.0, COL_W, [
        ("Sentinel-1 GRD  —  AWS Open Data mirror",
         "catalogue via CDSE OData · no account needed\n"
         "windowed /vsicurl reads: a scene costs 2 MB, not 1 GB"),
        ("Wind — ERA5  —  Open-Meteo archive",
         "the same reanalysis at the same 0.25° grid, over open HTTP\n"
         "cached per grid cell per day, so a re-run needs no network"),
        ("Currents — CMEMS  —  Copernicus Marine",
         "username and password; there is no API key\n"
         "one NetCDF subset per scene"),
        ("AIS positions  —  NOAA Marine Cadastre",
         "no account · 330 MB a day, streamed and filtered in flight\n"
         "down to a few hundred KB covering one scene"),
        ("Dark vessels  —  Global Fishing Watch",
         "free token · AIS gap events carrying the publisher's own\n"
         "intentionalDisabling assessment"),
        ("Incidents  —  NOAA IncidentNews + curated catalogue",
         "3,415 documented spills · corroboration only,\n"
         "never used as training labels"),
    ], SOURCE, fs=7.4, tfs=8.9, gap=2.6, align="left")

    down(SPINE, src[-1][1] - 0.5, src[-1][1] - 4.6, SOURCE)

    # ---- 2. pipeline -----------------------------------------------------
    top_pipe = src[-1][1] - 7.4
    band(top_pipe + 3.6, "2 · PIPELINE      runs once, during docker build", STAGE)
    pipe = stack(COL_X, top_pipe, COL_W, [
        ("ingest/",
         "calibrate to Sigma0 dB  ·  refined-Lee speckle (float32)\n"
         "land mask  ·  512 px tiles at ~80 m\n"
         "a lightly-filtered copy is kept alongside, because heavy\n"
         "filtering destroys the texture the next stage needs"),
        ("detect/stage_b   —   find the dark patches",
         "classical adaptive dark-patch detector.\n"
         "The trained U-Net is parked: 0.70 IoU on balanced patches,\n"
         "but it does not transfer to whole scenes."),
        ("detect/lookalike   —   but is it really oil?",
         "HARD GATES first, physics before any scoring:\n"
         "   wind < 2 m/s  → reject — calm water looks exactly like oil\n"
         "   wind > 15 m/s → reject — oil is mixed into the wave field\n"
         "   in between    → a GRADED penalty, not a gate\n"
         "then a small interpretable logistic model over damping, shape,\n"
         "texture and VH/VV. Every rejection can be explained out loud."),
        ("detect/wavetrain  +  morphology",
         "scene-level periodicity → an internal-wave train\n"
         "long and thin → a moving vessel   ·   blob → a fixed source"),
        ("registry cross-check",
         "a documented fixed source inside a DRIFT-SCALED radius routes the\n"
         "slick to infrastructure or natural seep BEFORE any vessel is\n"
         "considered. Blaming a ship for a known wellhead is the worst\n"
         "error this system can make."),
        ("drift/   —   backwards, to where it started",
         "RK4 advection on real CMEMS currents\n"
         "weathering disabled: those processes are not reversible\n"
         "output: an origin, an uncertainty radius, a particle track"),
        ("attribute/   —   who was there at the time?",
         "parity        how parallel the track is to the slick's long axis\n"
         "proximity     Gaussian on the drift uncertainty\n"
         "temporality   late arrivals decay three times faster\n"
         "plus a bonus when a vessel went dark at the origin"),
        ("decision/   —   rank, or decline to",
         "tiers: confirmed · probable · possible · insufficient\n"
         "ABSTAIN when the top two are within noise, or when no AIS\n"
         "covers the estimated origin at all"),
    ], STAGE, fs=7.5, tfs=9.4, arrows=True)

    # The three physics stages are the contribution; mark them as one block.
    top_phys, bot_phys = pipe[2][0], pipe[4][1]
    ax.add_patch(mp.FancyBboxPatch(
        (COL_X - 1.1, bot_phys - 1.1), COL_W + 2.2, (top_phys - bot_phys) + 2.2,
        boxstyle="round,pad=0.3,rounding_size=1.0", linewidth=1.7,
        edgecolor=PHYSICS, facecolor="none", linestyle=(0, (6, 3)), zorder=5))

    down(SPINE, pipe[-1][1] - 0.5, pipe[-1][1] - 5.2, SERVE, "the analysis result")

    # ---- 3. serving ------------------------------------------------------
    top_serve = pipe[-1][1] - 8.0
    band(top_serve + 3.6, "3 · SERVING      the container never runs the pipeline",
         SERVE)
    serve = stack(COL_X, top_serve, COL_W, [
        ("build-time precompute",
         "everything above runs during `docker build`, where memory is\n"
         "plentiful, and pickles only the RESULT: polygons, scores, drift\n"
         "tracks. 730 KB for 16 scenes, with no arrays at all."),
        ("world_index.json   —   the map's payload",
         "built once. Ages are recomputed on every request, because\n"
         "'active' means younger than 72 h, which is true only at the\n"
         "moment the cache was written."),
        ("api/   FastAPI",
         "serves cached results only. ALLOW_LIVE_ANALYSIS=false — the\n"
         "coastline grid alone is 933 MB against a 512 MB container.\n"
         "74 MB resident in production."),
        ("ui/   Leaflet map",
         "attributed (orange) · insufficient data (amber) · past (brown)\n"
         "documented incident (pink) · rejected look-alike (grey)\n"
         "the one synthetic scene is drawn dashed and labelled"),
    ], SERVE, fs=7.5, tfs=9.4, arrows=True)

    # ---- limits, running alongside the pipeline --------------------------
    band(top_pipe + 3.6, "STATED LIMITS", WARN, x=SIDE_X)
    stack(SIDE_X, top_pipe, SIDE_W, [
        ("not real-time",
         "imagery lands 3–24 h after acquisition\n"
         "free AIS lags about 72 h\n"
         "revisit over one point is 6–12 days"),
        ("what SAR cannot do",
         "it cannot measure oil thickness,\nvolume or type\n\n"
         "outside the wind window,\n'no detection' ≠ 'no oil'"),
        ("ranked, never accused",
         "candidates are CORRELATIONS between\na drift-estimated origin and AIS\n"
         "tracks: investigative leads, never\nevidence of responsibility"),
        ("running, not as designed",
         "segmentation is the classical detector,\nnot the trained U-Net\n\n"
         "look-alike weights are hand-set\nphysical priors, not fitted — the\n"
         "PANGAEA fit was attempted and\nabandoned on evidence"),
        ("one synthetic scene of sixteen",
         "15 are real Sentinel-1. One is\nfabricated, drawn dashed and labelled\n"
         "everywhere it appears, because no\nreal scene can demonstrate vessel\n"
         "ranking: no AIS coverage, or a\ndocumented fixed source, or no slick"),
        ("97% abstention is the system working",
         "1 of 34 detections carries a ranked\nvessel. The rest decline for want of\n"
         "evidence rather than guessing."),
    ], WARN, fs=7.3, tfs=8.8, gap=3.0)

    ax.text(3, serve[-1][1] - 6.0,
            "Every stage shown is what the code does today. Where the running system "
            "differs from the original plan, the box says so rather than showing the "
            "plan.", fontsize=8.6, color=MUTED, va="top", style="italic")
    ax.text(3, serve[-1][1] - 9.4,
            "Read top to bottom: data is fetched, the pipeline runs once at build "
            "time, and the container only ever deserialises the answer.",
            fontsize=8.6, color=MUTED, va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="jpeg", dpi=150, bbox_inches="tight",
                pil_kwargs={"quality": 92})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/architecture.jpg")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    draw(out)
    print(f"-> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
