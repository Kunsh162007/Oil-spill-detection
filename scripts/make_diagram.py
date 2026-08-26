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
    ax.text(3, 165.3, "How it works, end to end   ·   all data is real   ·   "
                      "oilspill.onrender.com", fontsize=10.4, color=MUTED, va="top")
    ax.plot([3, 97], [163.0, 163.0], color=LINE, linewidth=1.2)

    COL_X, COL_W = 3.0, 60.0
    SPINE = COL_X + COL_W / 2
    SIDE_X, SIDE_W = 67.0, 30.0

    # ---- 1. data in ------------------------------------------------------
    band(160.4, "1 · WHAT GOES IN      all downloaded from public sources, nothing made up", SOURCE)
    src = stack(COL_X, 156.0, COL_W, [
        ("Sentinel-1 GRD  —  AWS Open Data mirror",
         "Radar photos of the sea from a satellite. Free, no sign-up.\n"
         "We download only the small patch we need: 2 MB, not 1 GB."),
        ("Wind — ERA5  —  Open-Meteo archive",
         "How hard the wind was blowing there, at that exact time. Free.\n"
         "Wind is the single biggest clue for telling oil from look-alikes."),
        ("Currents — CMEMS  —  Copernicus Marine",
         "Which way the sea was flowing, so we can trace oil backwards.\n"
         "Needs a free account login."),
        ("AIS positions  —  NOAA Marine Cadastre",
         "Positions the ships broadcast about themselves. Free.\n"
         "A day is 330 MB; we keep only the ships near our patch of sea."),
        ("Dark vessels  —  Global Fishing Watch",
         "Ships that switched their tracker OFF. Free sign-up.\n"
         "A ship hiding is a stronger clue than one in plain sight."),
        ("Incidents  —  NOAA IncidentNews + curated catalogue",
         "A public list of 3,415 real, recorded oil spills.\n"
         "Used to double-check our answers, never to train the system."),
    ], SOURCE, fs=7.4, tfs=8.9, gap=2.6, align="left")

    down(SPINE, src[-1][1] - 0.5, src[-1][1] - 4.6, SOURCE)

    # ---- 2. pipeline -----------------------------------------------------
    top_pipe = src[-1][1] - 7.4
    band(top_pipe + 3.6, "2 · WHAT THE SYSTEM DOES      each step, in order", STAGE)
    pipe = stack(COL_X, top_pipe, COL_W, [
        ("ingest/",
         "Turn the raw satellite numbers into true brightness,\n"
         "blur out the speckly radar noise, cut out land, and slice\n"
         "the scene into tiles. We keep a less-blurred copy too,\n"
         "because blurring hides the fine detail the next step needs."),
        ("detect/stage_b   —   find the dark patches",
         "Oil flattens the sea, so it shows up as a DARK PATCH on radar.\n"
         "This step finds every dark patch. Many will not be oil.\n"
         "(A trained AI model exists but did not work on full scenes.)"),
        ("detect/lookalike   —   but is it really oil?",
         "Lots of things look like oil: calm patches, algae, rain, waves.\n"
         "Wind decides. Too calm (under 2 m/s) and flat sea looks\n"
         "identical to oil. Too rough (over 15 m/s) and oil is churned\n"
         "away. Between those, we score how oil-like the patch is:\n"
         "how dark, what shape, how smooth, how it looks on two radar\n"
         "channels. Every rejection comes with a plain-English reason."),
        ("detect/wavetrain  +  morphology",
         "Regular stripes across the scene = underwater waves, not oil.\n"
         "A long thin streak = a moving ship. A blob = something fixed."),
        ("registry cross-check",
         "Before blaming any ship, check the public spill list. If a known\n"
         "leaking wellhead or natural seep is nearby, the oil is credited\n"
         "to that instead. Blaming a passing ship for a known wellhead\n"
         "is the worst mistake this system could make."),
        ("drift/   —   backwards, to where it started",
         "Oil drifts after it spills, so where we SEE it is not where it\n"
         "STARTED. We run the sea currents backwards to estimate the\n"
         "starting point, plus how uncertain that estimate is."),
        ("attribute/   —   who was there at the time?",
         "Now look at ships near that starting point and score each one:\n"
         "was it heading the same way as the streak? how close did it\n"
         "pass? how recently? did it switch its tracker off right there?\n"
         "Each ship gets a score out of these four questions."),
        ("decision/   —   rank, or decline to",
         "Rank the ships and label our confidence in the oil itself.\n"
         "If the top two ships are too close to call, or there are no ship\n"
         "records at all, we SAY SO instead of guessing."),
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
    band(top_serve + 3.6, "3 · HOW THE WEBSITE SHOWS IT      the answers are worked out in advance",
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
    band(top_pipe + 3.6, "WHAT WE DO NOT CLAIM", WARN, x=SIDE_X)
    stack(SIDE_X, top_pipe, SIDE_W, [
        ("We cannot watch all the time",
         "The satellite passes over the same\n"
         "sea only every 6-12 days. Oil is gone\n"
         "in days. A ship could dump the morning\n"
         "after a pass and leave nothing to see.\n"
         "We take occasional photos, not video."),
        ("We cannot see it as it happens",
         "The photo reaches us 3-24 hours late.\n"
         "Free ship data is ~3 days behind.\n"
         "So: 'found soon after', never 'live'."),
        ("We cannot see oil in the wrong wind",
         "Radar does not see oil. It sees that the\n"
         "sea is unusually SMOOTH where oil is.\n"
         "Too calm: the sea is already smooth.\n"
         "Too rough: waves churn the oil under.\n"
         "So 'we found no oil' is NOT the same\n"
         "as 'there was no oil'."),
        ("We cannot tell how much oil",
         "No thickness, so no volume. A thin\n"
         "sheen and a bad spill can look alike.\n"
         "No oil type either - crude, diesel and\n"
         "cooking oil all just look dark."),
        ("We cannot prove who did it",
         "We find: oil probably started near here,\n"
         "around this time, these ships were near.\n"
         "That is a COINCIDENCE, not evidence.\n"
         "The drift guess is off by tens of km,\n"
         "busy lanes have many ships, and a\n"
         "guilty ship can switch its tracker off\n"
         "and never appear in our list at all.\n"
         "So: a ranked list, never one name."),
        ("We often cannot name anyone",
         "1 of 34 detections has ships ranked.\n"
         "The rest say 'not enough evidence'.\n"
         "A system that always gives a name\n"
         "is a system that is guessing."),
        ("We cannot see tiny or faint slicks",
         "Under ~0.05 km2 a dark patch cannot\n"
         "be told apart from radar noise.\n"
         "Small dumps can slip under that."),
        ("We are weaker near the coast",
         "Harbours, shallows and land create\n"
         "confusing dark and bright patterns."),
        ("What we have not finished",
         "The AI model works on small tiles but\n"
         "not full scenes, so a simpler method\n"
         "runs instead. The oil-vs-look-alike\n"
         "scores are set by hand from physics,\n"
         "not learned from data."),
        ("What radar CAN do, for balance",
         "It sees through cloud and works at\n"
         "night. That is why radar is used at all.\n"
         "These limits are the price of that."),
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
