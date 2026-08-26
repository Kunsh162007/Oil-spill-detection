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

    P("0. In plain words &mdash; read this first", "h1")
    P("Ships sometimes dump oily waste at sea and sail away. Nobody is watching "
      "most of the ocean, so it usually goes unnoticed. This system watches it "
      "from a satellite.", "p")
    P("<b>How it works, in four sentences.</b> A radar satellite photographs the "
      "sea. Oil flattens the water, so it shows up as a dark patch. Many other "
      "things also look dark &mdash; calm water, algae, rain &mdash; so we use the "
      "wind, the shape and the texture to work out which patches are really oil. "
      "Then, because oil drifts, we run the sea currents backwards to guess where "
      "it started, and list which ships were near that spot at that time.", "p")
    P("<b>The most important thing to understand.</b> We produce a <i>ranked list "
      "of ships that could be responsible</i>. That is a lead for an investigator "
      "to follow, not proof. A ship being near a spill is a coincidence until "
      "somebody checks. If the evidence is weak, the system says \"not enough "
      "evidence\" instead of guessing &mdash; and today it says that 97% of the "
      "time, which is it working properly, not failing.", "p")
    P("<b>Jargon you will hear.</b> "
      "<i>SAR</i> is the radar camera on the satellite. "
      "<i>Look-alike</i> is something dark that is not oil. "
      "<i>AIS</i> is the radio signal ships broadcast saying where they are. "
      "<i>Drift</i> is oil moving with the current after it spills. "
      "<i>Abstain</i> means we decline to name anyone.", "p")

    P("1. What the system does, in more detail", "h1")
    P("A Sentinel-1 radar scene of the ocean goes in. Dark patches come out, each "
      "tested against physics to decide whether it is oil or a natural look-alike. "
      "Surviving slicks are drifted backwards through real ocean currents to estimate "
      "where they started, and AIS tracks near that origin are scored to produce a "
      "<b>ranked list of candidate vessels</b> &mdash; never a single accusation.")
    P("Deployed at oilspill.onrender.com: 16 scenes, 34 presented detections, "
      "74 MB resident against a 512 MB limit.")

    P("2. Where the data comes from, and what we turned down", "h1")
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

    P("3. Which methods we used to find the oil", "h1")
    P("3.1 Finding the dark patches", "h2")
    P("The plan called for a U-Net with a ResNet-34 encoder, chosen because SkyTruth "
      "Cerulean uses it for this exact task &mdash; a complete answer to \"why this "
      "architecture\". It was trained, reached <b>0.70 oil IoU on balanced "
      "patches</b>, and then failed to transfer to whole scenes, where overlap with "
      "truth was near zero.")
    P("<b>What runs instead:</b> a classical adaptive dark-patch detector. The "
      "checkpoint is parked rather than deleted, and the running system reports "
      "<i>classical-dark-patch</i> on its own status panel rather than implying a "
      "deep model is active. A detector that works beats a better one that does not.")

    P("3.2 Deciding which dark patches are really oil", "h2")
    P("Deliberately <b>not</b> a neural network. A small logistic model over "
      "interpretable physical features means every rejection can be explained out "
      "loud: <i>\"rejected: wind 1.2 m/s, below the 2 m/s floor\"</i>. An explainable "
      "rejection is worth more here than a fractional IoU gain buried in a tensor.")
    P("Wind is kept <b>out</b> of the network for the same reason. Adding it as a "
      "fourth input channel would probably help slightly, and would make the "
      "wind-ablation claim unmeasurable.")

    P("4. The science rules we built in", "h1")
    P("4.1 Wind: a strict rule at one end, a flexible one at the other", "h2")
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

    P("4.2 The longer ago it spilled, the further away we look", "h2")
    P("A flat 60 km asked a slick found a fortnight after an incident to have stayed "
      "where it started, which is the one thing oil never does. Real MSC ELSA 3 "
      "detections 15 days later sat 87 km out and scored as uncorroborated. The "
      "radius is now 30 km plus 20 km per elapsed day, capped at 250 km &mdash; below "
      "the 0.3 m/s the project's own validation script quotes, so it errs toward "
      "matching too little rather than too much.")

    P("4.3 Never blame a ship for a leak that is already known about", "h2")
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

    P("5. How the website runs on a free server", "h1")
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

    P("6. What we changed our minds about, and why", "h1")
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
    P("7. A thing we tried that did not work, and why we are saying so", "h1")
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

    P("8. What we claim, and what we refuse to claim", "h1")
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

    F.append(PageBreak())
    P("9. Things this system cannot do", "h1")
    P("Written plainly, because these are the questions the system will be asked "
      "and it is better to answer them first. None of these are bugs. They are "
      "limits of radar, of free data, or of what a correlation can honestly "
      "support.", "p")

    P("9.1 We cannot watch continuously", "h2")
    P("The satellite is not parked over India. It circles the Earth, and it passes "
      "over any given patch of sea roughly <b>once every 6 to 12 days</b>. Oil "
      "spreads, breaks up and disappears in days. So a ship could dump oil the "
      "morning after a pass, and by the next pass there would be nothing left to "
      "see. <b>We are taking occasional photographs, not watching a video.</b> "
      "Anyone who says otherwise about a free satellite service is mistaken.", "p")

    P("9.2 We cannot see it as it happens", "h2")
    P("Even when the satellite does pass over, the picture takes <b>3 to 24 "
      "hours</b> to reach us: it has to be downlinked, processed and published. "
      "The free ship-tracking data is slower still, about <b>3 days</b> behind. So "
      "the honest description is \"we find it soon after\", never \"we find it "
      "live\". A spill we report this morning may have happened yesterday, and the "
      "ship may already be somewhere else.", "p")

    P("9.3 We cannot see oil in the wrong wind", "h2")
    P("This is the one people find most surprising. Radar does not see oil "
      "directly &mdash; it sees that the sea is unusually <i>smooth</i> where the "
      "oil is. That only works in a middle range of wind.", "p")
    P("If the wind is <b>too light</b> (under about 2 m/s), the whole sea is "
      "already smooth, so oil looks the same as everything around it. If the wind "
      "is <b>too strong</b> (above about 15 m/s), the waves churn the oil under "
      "and it stops flattening anything. Either way we see nothing.", "p")
    P("The consequence matters: <b>\"we found no oil\" is not the same as "
      "\"there was no oil\".</b> If the wind was wrong, we simply could not have "
      "seen it. The system records the wind with every result so this can always "
      "be checked.", "p")

    P("9.4 We cannot tell how much oil there is", "h2")
    P("Radar tells us <i>something is there</i> and roughly what shape and area it "
      "covers. It cannot tell us <b>how thick the layer is</b>, so it cannot tell "
      "us the volume. A very thin sheen and a serious slick can look similar. It "
      "also cannot tell us <b>what kind</b> of oil it is &mdash; crude, diesel, "
      "fuel oil or vegetable oil all just look dark. Only a ship or an aircraft "
      "taking a sample can answer those.", "p")

    P("9.5 We cannot prove who did it", "h2")
    P("This is the most important limit in the whole document. What the system "
      "actually finds is: <i>this oil probably started near here, around this "
      "time, and these ships were nearby.</i> That is a <b>coincidence in space "
      "and time</b>. It is a good reason for an investigator to look closer. It is "
      "not evidence.", "p")
    P("Three things stop it from being proof. The backwards drift estimate has "
      "real uncertainty &mdash; typically tens of kilometres, and it grows the "
      "further back we go. Busy sea lanes have many ships passing the same point. "
      "And ship tracking can be switched off or falsified, so the guilty ship may "
      "simply not appear in the list at all. <b>We therefore publish a ranked list "
      "with confidence scores, and never name a single culprit.</b>", "p")

    P("9.6 We often cannot name anyone at all", "h2")
    P("Right now <b>1 of our 34 detections</b> has any ships ranked against it. "
      "For the other 33 the system says \"not enough evidence\". That is "
      "deliberate. The reasons are: free ship data does not cover Indian waters in "
      "our sample; some slicks come from a known leaking wellhead where blaming a "
      "passing ship would be plainly wrong; and some had no ship anywhere near the "
      "estimated origin. <b>A system that always produces a name is a system that "
      "is guessing.</b>", "p")

    P("9.7 We cannot see very small or very faint slicks", "h2")
    P("Below roughly 0.05 km&sup2; a dark patch is indistinguishable from ordinary "
      "radar noise, so it is discarded. Small deliberate discharges can slip under "
      "that floor. Raising the sensitivity would mean reporting noise as oil, "
      "which is worse.", "p")

    P("9.8 We cannot work reliably close to shore", "h2")
    P("Near the coast, harbours, shallow water and land features all create dark "
      "and bright patterns that confuse the detector. Results in coastal water are "
      "weaker than in open sea, and our test data for coastal cases is thin &mdash; "
      "19 samples out of 174.", "p")

    P("9.9 What we have not finished", "h2")
    P("Two parts of the system are not what was planned, and the interface says so "
      "rather than hiding it. The AI segmentation model was trained and reached "
      "usable accuracy on small image tiles, but did not work on full scenes, so a "
      "simpler and more reliable method is running instead. And the oil-versus-"
      "look-alike scoring uses values set by hand from known physics rather than "
      "learned from data &mdash; we tried to learn them from a public dataset and "
      "stopped when we found that dataset could not support it honestly (section "
      "7).", "p")

    P("9.10 What radar CAN do, for balance", "h2")
    P("It sees through cloud, and it works at night. That is why radar is used for "
      "this at all: an ordinary camera would be blind half the time and blocked by "
      "weather the rest. The limits above are the price of that capability, not "
      "signs of a broken system.", "p")

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
