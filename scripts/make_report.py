"""Build the decision record as a PDF.

    python scripts/make_report.py --out docs/decision-record.pdf

Every choice is recorded with the alternative that was rejected and the reason.
Where a decision was later reversed, the reversal is recorded too - a document
listing only what survived reads as though nothing was learned.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

INK = "#111820"
ACCENT = "#1f6f8b"
MUTED = "#5b6b7a"


def styles():
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle("t", parent=base["Title"], fontSize=22, leading=26,
                                textColor=INK, alignment=0, spaceAfter=4)
    s["sub"] = ParagraphStyle("s", parent=base["Normal"], fontSize=10.5, leading=14,
                              textColor=MUTED, spaceAfter=18)
    s["h1"] = ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, leading=17,
                             textColor=ACCENT, spaceBefore=16, spaceAfter=7)
    s["h2"] = ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, leading=14,
                             textColor=INK, spaceBefore=10, spaceAfter=4)
    s["p"] = ParagraphStyle("p", parent=base["Normal"], fontSize=9.6, leading=13.6,
                            textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7)
    s["small"] = ParagraphStyle("sm", parent=base["Normal"], fontSize=8.6,
                                leading=11.6, textColor=MUTED, spaceAfter=6)
    s["cell"] = ParagraphStyle("c", parent=base["Normal"], fontSize=8.4, leading=11)
    s["cellb"] = ParagraphStyle("cb", parent=base["Normal"], fontSize=8.4, leading=11,
                                textColor=ACCENT, fontName="Helvetica-Bold")
    return s


def table(rows, widths, s, header=True):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    data = []
    for i, row in enumerate(rows):
        style = s["cellb"] if (header and i == 0) else s["cell"]
        data.append([Paragraph(str(c), style) for c in row])
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor(ACCENT)),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#dde3e9")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def build(out_path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    s = styles()
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=18 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Oil Spill Detection - Decision Record")
    W = doc.width
    F = []

    def P(text, k="p"):
        F.append(Paragraph(text, s[k]))

    P("Oil Spill Detection &amp; Vessel Attribution", "title")
    P("Decision record &mdash; what was chosen, what was rejected, and what changed. "
      "Generated 26 August 2026.", "sub")

    P("1. What the system does", "h1")
    P("A Sentinel-1 radar scene of the ocean goes in. Dark patches come out, each "
      "tested against physics to decide whether it is oil or a natural look-alike. "
      "Surviving slicks are drifted backwards through real ocean currents to estimate "
      "where they started, and AIS tracks near that origin are scored to produce a "
      "<b>ranked list of candidate vessels</b> &mdash; never a single accusation.")
    P("Deployed at oilspill.onrender.com: 16 scenes, 34 presented detections, "
      "74 MB resident against a 512 MB limit.")

    P("2. Data sources: what was chosen and what was not", "h1")
    P("The binding constraint was access, not quality. Every source had to be "
      "obtainable without an institutional affiliation, and two datasets the original "
      "plan depended on turned out not to be.")
    F.append(table([
        ["Need", "Chosen", "Rejected, and why"],
        ["SAR imagery",
         "<b>AWS Open Data mirror</b>, catalogue via CDSE OData. Windowed "
         "<i>/vsicurl</i> reads: a scene costs 2 MB, not 1 GB.",
         "<b>CDSE download</b> needs an account whose registration page would not "
         "render. <b>Planetary Computer</b> &mdash; Hub deprecated."],
        ["Wind",
         "<b>ERA5 via Open-Meteo</b>. Same reanalysis, same 0.25&deg; grid, open "
         "HTTP, no key.",
         "<b>Copernicus CDS</b> &mdash; identical data, but blocks on registration "
         "and a <i>.cdsapirc</i>."],
        ["Currents", "<b>Copernicus Marine (CMEMS)</b>, one NetCDF per scene.",
         "No open alternative at this resolution. Needs a username and password; "
         "there is no API key, which is the usual confusion."],
        ["AIS positions",
         "<b>NOAA Marine Cadastre</b> &mdash; the only free source here publishing "
         "real position reports.",
         "<b>GFW</b> serves identity and events but <i>no tracks</i>. <b>Danish "
         "DMA</b> host unreachable. <b>AISStream</b> is live-only, so it cannot match "
         "a past scene. <b>MarineTraffic / Spire</b> are paid."],
        ["Dark vessels",
         "<b>GFW gap events</b>, carrying the publisher's own "
         "<i>intentionalDisabling</i> assessment.",
         "Nothing else publishes an adjudicated AIS switch-off. Coverage is thin: one "
         "gap in the whole Indian Ocean box over ten days."],
        ["Incidents",
         "<b>NOAA IncidentNews</b> plus a curated world catalogue, 3,415 spills.",
         "Used for corroboration only, never as training labels."],
    ], [W * 0.13, W * 0.40, W * 0.47], s))

    P("3. Model choices", "h1")
    P("3.1 Segmentation", "h2")
    P("The plan called for a U-Net with a ResNet-34 encoder, chosen because SkyTruth "
      "Cerulean uses it for this exact task &mdash; a complete answer to \"why this "
      "architecture\". It was trained, reached <b>0.70 oil IoU on balanced "
      "patches</b>, and then failed to transfer to whole scenes, where overlap with "
      "truth was near zero.")
    P("<b>What runs instead:</b> a classical adaptive dark-patch detector. The "
      "checkpoint is parked rather than deleted, and the running system reports "
      "<i>classical-dark-patch</i> on its own status panel rather than implying a "
      "deep model is active. A detector that works beats a better one that does not.")

    P("3.2 Look-alike rejection", "h2")
    P("Deliberately <b>not</b> a neural network. A small logistic model over "
      "interpretable physical features means every rejection can be explained out "
      "loud: <i>\"rejected: wind 1.2 m/s, below the 2 m/s floor\"</i>. An explainable "
      "rejection is worth more here than a fractional IoU gain buried in a tensor.")
    P("Wind is kept <b>out</b> of the network for the same reason. Adding it as a "
      "fourth input channel would probably help slightly, and would make the "
      "wind-ablation claim unmeasurable.")

    P("4. Physics decisions", "h1")
    P("4.1 The wind window is hard at the bottom, graded at the top", "h2")
    P("The two ends of the detection window fail differently, and treating them "
      "identically was a bug. <b>Below</b> the floor, calm water is genuinely "
      "indistinguishable from oil &mdash; every dark patch is unexplainable, so a "
      "hard gate is right and it prevents false positives. <b>Above</b> the ceiling "
      "the failure inverts: oil becomes invisible, so a large, strongly damped, "
      "elongated feature seen anyway is <i>evidence</i>, and gating it converts a "
      "real detection into a miss.")
    P("The ceiling is now a graded penalty with a hard gate only at 15 m/s. This is "
      "what the brief asked for &mdash; \"soft edges &hellip; a graded feature, not a "
      "hard cutoff\" &mdash; and the original code did not implement it.")

    P("4.2 Corroboration radius scales with drift time", "h2")
    P("A flat 60 km asked a slick found a fortnight after an incident to have stayed "
      "where it started, which is the one thing oil never does. Real MSC ELSA 3 "
      "detections 15 days later sat 87 km out and scored as uncorroborated. The "
      "radius is now 30 km plus 20 km per elapsed day, capped at 250 km &mdash; below "
      "the 0.3 m/s the project's own validation script quotes, so it errs toward "
      "matching too little rather than too much.")

    P("4.3 Fixed sources are excluded before vessel attribution", "h2")
    P("The most serious bug found in this build. On the Taylor Energy MC-20 scene the "
      "system ranked three named vessels &mdash; FAST RUNNER, SM NEW ORLEANS, GULF "
      "DAWN &mdash; against oil from a wellhead that has leaked continuously since "
      "2004. One did not even abstain. The brief calls this the worst failure the "
      "system can produce.")
    P("Two independent gaps allowed it. Morphology recognised a known fixed source "
      "only within 15 km, and the plume streams well past that. And the registry, "
      "which <i>had</i> identified the source correctly on every one of them, was "
      "never consulted for the decision, because <tt>SpillIncident.to_dict</tt> "
      "dropped the flag before it reached the pipeline. A documented fixed-source "
      "match now overrides the geometric test: an independent identification outranks "
      "a radius.")

    P("5. Serving architecture", "h1")
    P("The free tier gives 512 MB. The pipeline needs far more, but the <i>result</i> "
      "is tiny &mdash; 730 KB of polygons and scores for 16 scenes, with no arrays at "
      "all. So the analysis runs once during <tt>docker build</tt>, where memory is "
      "plentiful, and the container only ever deserialises the answer.")
    P("<b>Rejected:</b> analysing per request (OOM-killed mid-request, which surfaces "
      "as a 502 with no body and no log line); a larger paid instance (unnecessary "
      "once the real memory bugs were found &mdash; one import alone was allocating "
      "933 MB); a self-managed Oracle VM (trades a <tt>git push</tt> deploy for "
      "server administration the night before a demo).")
    P("The container also refuses live analysis outright. Without that, one "
      "\"re-analyse\" click loads the 933 MB coastline grid and kills the service.")

    P("6. What changed during the build, and why", "h1")
    F.append(table([
        ["Change", "Why"],
        ["Wind moved from a hardcoded constant to real ERA5",
         "Both configs used a climatological constant sitting mid-window, so every "
         "candidate scored a perfect wind match and the strongest discriminator "
         "contributed nothing. <b>63 detections became 28.</b> Most of the loss was "
         "MSC ELSA 3, where real wind was 11.4&ndash;13.4 m/s: those detections "
         "existed only because the model was fed an invented 6.1 m/s."],
        ["ELSA 3 validated on a different scene",
         "The 28 May pass is outside the wind window. A 9 June pass &mdash; same "
         "satellite, same orbit, 15 days after the sinking with the wreck still "
         "leaking, ERA5 wind 5.33 m/s &mdash; supports the claim honestly. All four "
         "validation checks now pass."],
        ["Currents moved to real CMEMS",
         "With a synthetic field the drift-direction check was close to a coin flip, "
         "as the validation script said in its own output. Real currents produce "
         "about three times the correction, both toward the wreck."],
        ["Drift interpolation rewritten",
         "<tt>xarray.interp</tt> rebuilt an intermediate Dataset per particle per "
         "timestep: 11.9 ms a sample, and one scene took <b>620 s</b> against a "
         "two-minute target. Preloading into numpy made it 22 &micro;s &mdash; "
         "<b>4.6 s</b> for the same scene, with a maximum disagreement of "
         "2&times;10<sup>-9</sup> m/s against the old path."],
        ["Unattributable detections split out of \"active\"",
         "Of 26 current detections only one had a ranked vessel. Listing the other 25 "
         "alongside it implied leads that do not exist; they now sit under "
         "<i>Insufficient data</i>."],
        ["Synthetic scenes removed, then one restored and labelled",
         "With 15 real scenes a fabricated one is a liability. But no real scene can "
         "demonstrate vessel ranking &mdash; no AIS, or a fixed source, or no slick "
         "&mdash; so exactly one returns, drawn dashed and labelled everywhere."],
        ["Abstention rate corrected from 100% to 97.1%",
         "It counted the 211 rejected look-alikes as abstentions. A rejected "
         "candidate has no vessel by definition, so the denominator was swamped and "
         "the figure contradicted the map."],
    ], [W * 0.30, W * 0.70], s))

    F.append(PageBreak())
    P("7. The look-alike fit: a negative result", "h1")
    P("The brief names look-alike false-positive rate as the primary decider, and the "
      "model ships hand-set weights. Fitting them on Yang et al. (PANGAEA "
      "10.1594/PANGAEA.980773) would have turned a process claim into an accuracy "
      "claim. It was attempted and abandoned on evidence.")
    P("Three assumptions failed first. The <b>images are gated</b> &mdash; the bulk "
      "archive answers 401 and refers you to the principal investigator, exactly the "
      "situation the dataset had been chosen to avoid. There are <b>no k-means "
      "cluster labels</b>; the only subgroup split published is oil/no-oil &times; "
      "coast/water. What is open is the tabular metadata, and it names the source "
      "Sentinel-1 product for all 5,515 objects &mdash; so the extractor reads the "
      "<i>original calibrated imagery</i> off the AWS mirror instead of 8-bit JPGs, "
      "which is better than the dataset as distributed.")
    P("The blocker is annotation convention. Yang boxes oil objects but labels whole "
      "patches for look-alikes &mdash; reasonably, since there is nothing to box when "
      "the whole scene is the phenomenon. Median area is <b>1.1 km&sup2; for oil "
      "against 153.2 for look-alikes</b>, so <tt>log_area</tt> carries the strongest "
      "separation in the table while encoding nothing but annotation style. "
      "<tt>damping_db</tt> separates at <b>0.00</b>: a whole-patch label has no "
      "surrounding sea to compare against. A model fitted on that scores well on "
      "hold-out and is worse in production, so no model file was written.")
    P("<b>One result survived.</b> Wind is derived from position and timestamp, so "
      "the annotation asymmetry cannot touch it. Across 174 independently-labelled "
      "objects, look-alikes sit below the detection window <b>32% of the time against "
      "21% for real oil</b> &mdash; roughly 1.5&times;. That is direct support for the "
      "premise the whole differentiator rests on, and it depends on no part of our "
      "own detector. Full write-up: <tt>docs/lookalike-fit-attempt.md</tt>.")

    P("8. What is claimed, and what is not", "h1")
    F.append(table([
        ["Defensible", "Explicitly not claimed"],
        ["Near-real-time detection, 3&ndash;24 h after acquisition",
         "Real-time. Imagery lands 3&ndash;24 h late and free AIS lags ~72 h."],
        ["Probabilistic <i>ranked</i> attribution with an evidence bundle",
         "Identifying the responsible vessel. Correlation is not evidence."],
        ["Dark-vessel flagging from adjudicated AIS gaps",
         "Continuous monitoring. Revisit over one point is 6&ndash;12 days."],
        ["Every rejection explained in physical terms",
         "Oil thickness, volume or type. SAR cannot measure them."],
        ["Validation against a real, documented Indian incident",
         "A benchmark score. The look-alike model is not fitted, and the segmentation "
         "running is not the trained network."],
    ], [W * 0.48, W * 0.52], s))

    P("9. Known limitations", "h1")
    P("&bull; Segmentation is the classical detector; the trained U-Net does not "
      "transfer to whole scenes.<br/>"
      "&bull; Look-alike weights are hand-set physical priors, not fitted.<br/>"
      "&bull; Attribution abstains on 97% of detections &mdash; correctly: the real "
      "scenes have no AIS coverage, are documented fixed sources, or contain no "
      "slick.<br/>"
      "&bull; One of 16 scenes is synthetic, labelled as such wherever it appears."
      "<br/>"
      "&bull; Outside the wind window, \"no detection\" does not mean \"no oil\".")

    # The architecture diagram is a separate deliverable, not an appendix:
    # it is a tall top-to-bottom flow and shrinking it onto A4 made it
    # unreadable, which defeats the point of having drawn it.
    P("The architecture diagram is a separate file: "
      "<tt>docs/architecture.jpg</tt>.", "small")

    doc.build(F)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/decision-record.pdf")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out)
    print(f"-> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
