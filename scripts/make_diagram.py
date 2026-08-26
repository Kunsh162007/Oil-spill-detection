"""Render the current architecture as a JPEG flowchart.

    python scripts/make_diagram.py --out docs/architecture.jpg

Drawn from the code as it stands, not from the original design. Several stages
moved or changed during the build, and a diagram showing the intended pipeline
rather than the running one is worse than none. Where a stage is not what the
plan called for - the classical detector standing in for the trained U-Net, the
hand-set priors standing in for a fitted look-alike model - the box says so.

Boxes size themselves to their text and stack from a running cursor. The first
version hard-coded heights and every box whose body ran a line long spilled
through its own border.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

INK = "#0f1720"
MUTED = "#5b6b7a"
LINE = "#c2ccd6"
SOURCE = "#1f6f8b"      # external data
STAGE = "#26413c"       # pipeline
PHYSICS = "#8a4b1e"     # the differentiator
SERVE = "#3b3663"       # serving
WARN = "#9a2f2f"        # runs, but not as designed

LINE_H = 1.70           # vertical space one body line needs, in axis units
HEAD_H = 3.7            # title plus the gap beneath it
PAD_H = 1.6             # breathing room at the bottom
GAP = 2.4               # space between stacked boxes


def draw(out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(18.5, 12.6), dpi=165)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#fbfaf7")

    def height(body: str) -> float:
        lines = body.count("\n") + 1 if body else 0
        return HEAD_H + lines * LINE_H + PAD_H

    def box(x, top, w, title, body="", colour=STAGE, fs=7.4, tfs=9.0):
        """Draw a box whose TOP edge is at `top`. Returns its bottom edge."""
        h = height(body)
        bottom = top - h
        for fill, edge, z in ((colour, "none", 2), ("none", colour, 3)):
            ax.add_patch(mp.FancyBboxPatch(
                (x, bottom), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                linewidth=1.25, edgecolor=edge, facecolor=fill,
                alpha=.055 if z == 2 else 1.0, zorder=z))
        ax.text(x + w / 2, top - 1.25, title, ha="center", va="top",
                fontsize=tfs, color=colour, fontweight="bold", zorder=4)
        if body:
            ax.text(x + w / 2, top - HEAD_H, body, ha="center", va="top",
                    fontsize=fs, color=INK, linespacing=1.5, zorder=4)
        return bottom

    def stack(x, top, w, items, colour, fs=7.4, tfs=9.0, gap=GAP, arrows=False):
        """Lay boxes down a column, returning each one's (top, bottom)."""
        spans = []
        cursor = top
        for title, body in items:
            bottom = box(x, cursor, w, title, body, colour, fs, tfs)
            spans.append((cursor, bottom))
            cursor = bottom - gap
        if arrows:
            for (_, bottom), (nxt_top, _) in zip(spans, spans[1:]):
                ax.annotate("", xy=(x + w / 2, nxt_top), xytext=(x + w / 2, bottom),
                            zorder=1, arrowprops=dict(arrowstyle="-|>", color=LINE,
                                                      linewidth=1.5))
        return spans

    def arrow(x1, y1, x2, y2, colour=LINE, label="", rad=0.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.1,
                                    alpha=0.5,          # wiring, not content
                                    connectionstyle=f"arc3,rad={rad}",
                                    shrinkA=2, shrinkB=2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.6, label, ha="center",
                    va="bottom", fontsize=6.9, color=MUTED, style="italic", zorder=4)

    # ---- header ----------------------------------------------------------
    ax.text(2, 98.2, "Oil Spill Detection & Vessel Attribution",
            fontsize=19, color=INK, fontweight="bold", va="top")
    ax.text(2, 94.6, "Architecture as deployed   ·   every input real   ·   "
                     "oilspill.onrender.com", fontsize=9.8, color=MUTED, va="top")
    ax.plot([2, 98], [92.8, 92.8], color=LINE, linewidth=1.1)

    for x, label, colour in ((1.5, "EXTERNAL DATA", SOURCE),
                             (20.5, "PIPELINE   (runs once, at build time)", STAGE),
                             (62.5, "SERVING", SERVE),
                             (81.5, "STATED LIMITS", WARN)):
        ax.text(x, 91.0, label, fontsize=8.4, color=colour,
                fontweight="bold", va="top")

    TOP = 88.4

    # ---- external data ---------------------------------------------------
    sources = stack(1.5, TOP, 17, [
        ("Sentinel-1 GRD", "AWS Open Data mirror\ncatalogue: CDSE OData\nno account · windowed /vsicurl"),
        ("Wind — ERA5", "Open-Meteo archive · no account\ncached per 0.25° cell per day"),
        ("Currents — CMEMS", "Copernicus Marine\nuser+password, no API key\none NetCDF per scene"),
        ("AIS positions", "NOAA Marine Cadastre · no account\n330 MB/day, streamed and\nfiltered in flight to a few hundred KB"),
        ("Dark vessels", "Global Fishing Watch · free token\ngap events, intentionalDisabling"),
        ("Incidents", "NOAA IncidentNews + curated\nworld catalogue · 3,415 spills"),
    ], SOURCE, fs=7.2, tfs=8.6)

    # ---- pipeline --------------------------------------------------------
    detect_col = stack(20.5, TOP, 19, [
        ("ingest/", "calibrate → Sigma0 dB\nrefined-Lee speckle (float32)\nland mask · 512 px tiles\nlightly-filtered copy kept for texture"),
        ("detect/stage_b", "classical adaptive dark-patch.\nThe trained U-Net is parked: 0.70 IoU\non patches, but does not transfer\nto whole scenes."),
        ("detect/lookalike   ← differentiator",
         "HARD GATES, physics before scoring:\nwind < 2 m/s → reject (calm water is\nindistinguishable from oil)\nwind > 15 m/s → reject\nbetween → GRADED penalty, not a gate\nthen an interpretable logistic model"),
        ("wavetrain + morphology",
         "scene-level periodicity → internal waves\nlinear → vessel · blob → fixed source"),
    ], STAGE, arrows=True)

    attrib_col = stack(41.5, TOP, 19, [
        ("registry cross-check",
         "documented fixed source inside a\nDRIFT-SCALED radius → infrastructure\nor natural seep, BEFORE any vessel"),
        ("drift/  backward",
         "RK4 advection on real CMEMS currents\nweathering disabled (irreversible)\norigin + uncertainty + particle track"),
        ("attribute/",
         "parity (undirected axis) · proximity\n(Gaussian on drift σ) · temporality\n(late arrivals decay 3× faster)\n+ AIS-gap dark-vessel bonus"),
        ("decision/",
         "tiers: confirmed / probable /\npossible / insufficient\nABSTAIN when the top two are within\nnoise, or no AIS covers the origin"),
    ], STAGE, arrows=True)

    pipe = detect_col + attrib_col
    # Detection column feeds attribution.
    ax.annotate("", xy=(41.5, detect_col[0][0] - 2.5),
                xytext=(39.5, detect_col[-1][1] - 1.5), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=LINE,
                                linewidth=1.6,
                                connectionstyle="arc3,rad=-0.4"))

    # ---- serving ---------------------------------------------------------
    serve = stack(62.5, TOP, 17, [
        ("build-time precompute",
         "the whole pipeline runs during\n`docker build`, where memory is\nplentiful, and pickles only the RESULT:\npolygons, scores, tracks.\n730 KB for 16 scenes, no arrays."),
        ("world_index.json",
         "the map's payload, built once.\nAges are recomputed per request:\n'active' means < 72 h, which is true\nonly when the cache was written."),
        ("api/  FastAPI",
         "serves cached results only.\nALLOW_LIVE_ANALYSIS=false — the\ncoastline grid alone is 933 MB against\na 512 MB container.\n74 MB resident in production."),
        ("ui/  Leaflet",
         "attributed (orange) · insufficient\ndata (amber) · past (brown) ·\ndocumented (pink) · rejected (grey)\nsynthetic scene dashed and labelled"),
    ], SERVE, arrows=True)

    # ---- stated limits ---------------------------------------------------
    stack(81.5, TOP, 17, [
        ("not real-time",
         "imagery lands 3–24 h after\nacquisition; free AIS lags ~72 h;\nrevisit over one point is 6–12 days.\n\nSAR cannot measure oil thickness,\nvolume or type.\n\nOutside the wind window,\n'no detection' ≠ 'no oil'.\n\nRanked vessels are CORRELATIONS:\ninvestigative leads, never\nevidence of responsibility."),
        ("running, not as designed",
         "segmentation: classical detector,\nnot the trained U-Net.\n\nlook-alike weights: hand-set physical\npriors, not fitted — the PANGAEA fit\nwas attempted and abandoned on\nevidence. See the PDF."),
        ("one synthetic scene",
         "15 scenes are real Sentinel-1.\nOne is fabricated, drawn dashed and\nlabelled everywhere it appears,\nbecause no real scene can show\nvessel ranking: no AIS, or a fixed\nsource, or no slick."),
    ], WARN, fs=7.2, tfs=8.6)

    # ---- wiring ----------------------------------------------------------
    def mid(span):
        return (span[0] + span[1]) / 2

    arrow(18.7, mid(sources[0]), 20.5, mid(detect_col[0]), SOURCE)
    arrow(18.7, mid(sources[1]), 20.5, mid(detect_col[2]), SOURCE, "wind", 0.10)
    arrow(18.7, mid(sources[2]), 41.5, mid(attrib_col[1]), SOURCE, "currents", 0.10)
    arrow(18.7, mid(sources[3]), 41.5, mid(attrib_col[2]), SOURCE, "AIS", 0.08)
    arrow(18.7, mid(sources[4]), 41.5, mid(attrib_col[2]) - 2.5, SOURCE, "gaps", 0.06)
    arrow(18.7, mid(sources[5]), 41.5, mid(attrib_col[0]), SOURCE, "registry", -0.10)
    arrow(60.5, mid(attrib_col[3]), 62.5, mid(serve[0]), SERVE, "analysis result", -0.2)

    ax.text(1.5, 6.4,
            "Every stage shown is what the code does today. Where the running system "
            "differs from the original plan, the box says so rather than showing the plan.",
            fontsize=8.2, color=MUTED, va="top", style="italic")
    ax.text(1.5, 3.4,
            "Data flows left to right. The pipeline runs once at build time; the "
            "container only ever deserialises the answer.",
            fontsize=8.2, color=MUTED, va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="jpeg", dpi=165, bbox_inches="tight",
                pil_kwargs={"quality": 93})
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
