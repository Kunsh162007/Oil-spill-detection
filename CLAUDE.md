# CLAUDE.md — Oil Spill Detection & Vessel Attribution

**SIH 2026 · SIH26143 · NTRO · Theme: Space Technology**

`docs/decision-record.pdf` is the plain-language explanation of the whole project, written for technical and non-technical readers alike, and `docs/architecture.jpg` is the same thing as a flowchart. This file is the engineering contract.

---

## WHERE THE PROJECT STANDS  (26 Aug 2026)

Live at **oilspill.onrender.com** — 16 scenes, 34 presented detections,
74 MB resident against a 512 MB free tier.

**Every input is real.** Sentinel-1 from the AWS Open Data mirror (no account),
ERA5 wind via Open-Meteo (no account), CMEMS currents (login), NOAA Marine
Cadastre AIS positions (no account), GFW gap events (free token), NOAA
IncidentNews + a curated catalogue (3,415 spills). One of the 16 scenes is
synthetic, drawn dashed and labelled everywhere it appears, because no real
scene can demonstrate vessel ranking.

### Built and working
`core/contracts.py` frozen · `ingest/` · `detect/stage_b` (classical) ·
`detect/lookalike` with hard physics gates · `detect/wavetrain` ·
`detect/morphology` · registry corroboration · `drift/` backward on real
currents · `attribute/` AIS scoring + dark vessels · `decision/` tiers and
abstention · `api/` + `ui/` · build-time precompute · 230+ tests passing.

### Not what the plan called for — say so out loud
- **Segmentation is the classical dark-patch detector.** The trained U-Net
  reached 0.70 oil IoU on balanced patches and did not transfer to whole
  scenes. Checkpoint parked as `stage_b_patchonly.pt`.
- **Look-alike weights are hand-set physical priors, not fitted.** The PANGAEA
  fit was attempted and abandoned on evidence — see
  `docs/lookalike-fit-attempt.md`. The one result that survived: look-alikes
  sit below the wind window 32% of the time against 21% for real oil.

### The training data is gone; the checkpoint is not
Fine-tuning is finished, so the inputs to it were deleted (26 Aug 2026): the
Zenodo Gulf of Mexico patches under `data/raw/extracted/`, the
`data/dev/*_manifest.json` indices built from them, the `data/dev/scenes/`
GOM stubs, and the 2.4 MB PANGAEA tabular export at `data/raw/pangaea/`.

What survives is what the work produced, not what fed it:
`models/stage_b_patchonly.pt` (the checkpoint), `data/dev/lookalike_features.csv`
(the extracted features, tracked in git) and `docs/lookalike-fit-attempt.md`.
All of the deleted inputs are re-downloadable without credentials — Zenodo
record 4672426, and the PANGAEA curl printed by `scripts/extract_features.py`.
`configs/train_gom.yaml` still points at the manifests and says so at the top;
run `scripts/prepare_dataset.py` to rebuild them before training again.

### The corrections that mattered most
1. **Real wind cut detections from 63 to 28.** Both configs had used a
   climatological constant sitting mid-window, so every candidate scored a
   perfect wind match and the strongest discriminator contributed nothing.
2. **ELSA 3 revalidated on the 9 June 2025 pass** (ERA5 wind 5.33 m/s). The
   28 May scene is outside the window at 11.4–13.4 m/s. All four checks pass.
3. **Fixed sources now outrank geometry.** MC-20 was ranking three named
   vessels against a documented wellhead — rule 4's worst failure — because
   morphology only looked within 15 km and `to_dict` dropped the registry flag.
4. **The wind ceiling is graded, not a gate.** Low wind causes false positives
   and is gated hard; high wind causes false negatives, so a strong feature
   seen anyway is evidence. Hard gate only at 15 m/s.
5. **Corroboration radius scales with drift time**: 30 km + 20 km/day, capped
   at 250 km. A flat 60 km asked a fortnight-old slick to have stayed put.
6. **Drift interpolation: 620 s → 4.6 s** by preloading the field into numpy
   instead of calling `xarray.interp` per particle per timestep.
7. **Memory**: `global_land_mask` allocates 933 MB at import — it is now lazy.
   That, not the pipeline, was what kept OOM-killing the container.

### Known gaps
Attribution abstains on 97% of detections — correctly: no AIS coverage in
Indian waters, documented fixed sources, or no slick. The bake-off has latency
numbers but no accuracy numbers. Coastal test data is thin (19 of 174).

---

## THE TASK

Take a Sentinel-1 radar scene of the ocean. Find oil slicks. Decide whether each dark patch is really oil or a natural look-alike. Drift the slick backwards through ocean currents to estimate where it started. Ask which ship was there at that time.

Output per slick: polygon, confidence, source type, and a **ranked list of candidate vessels** — never a single accusation.

---

## HARD RULES

1. **Ranked candidates, never a single accusation.** Correlation is not evidence. Every output surface says so.
2. **Never say "real-time."** Imagery arrives 3–24 h after acquisition; free AIS is ~72 h delayed. Say "near-real-time detection; attribution as AIS becomes available."
3. **No detection without a wind check.** SAR oil detection works only in a moderate wind window — published ranges put the lower bound at 2–3 m/s and the upper at 7–12 m/s, with soft edges (oil is occasionally visible outside it). Below the lower bound, calm water is indistinguishable from oil; above the upper bound, oil mixes into the wave field. A candidate without wind context is not a candidate. Treat the window as a graded feature, not a hard cutoff.
4. **Exclude natural seeps and fixed infrastructure before vessel attribution.** Accusing a ship of a natural seep is the worst failure this system can produce.
5. **Abstain when uncertain.** Wind outside window, no AIS coverage, top-two candidates within noise → return "insufficient evidence."
6. **Never invent data.** Missing scene, empty API response, failed drift run → raise. Never substitute plausible values.
7. **Never train on Cerulean outputs as ground truth.** They are probable sources, not confirmed. Use as weak labels, tagged as such.

---

## PHYSICS YOU NEED TO KNOW

Radar pulses hit the sea. Wind roughens the surface → bright return. Oil damps small waves → **dark patch**.

The whole problem is that many things make dark patches:

| Look-alike | How to tell it apart |
|---|---|
| Low wind / wind shadow | Wind below ~2–3 m/s at that place and time |
| Algal bloom, biogenic film | Irregular edge, lower damping, different VH/VV |
| Rain cell | Distinct texture, often circular |
| Internal waves | Regular periodic banding |
| Ship wake | Narrow, straight, attached to a vessel |
| Upwelling / front | Large scale, follows ocean structure |

**Features to compute per candidate region:**
- Wind speed at that pixel/time (ERA5) — strongest single feature
- Shape: elongation, compactness, perimeter/area
- Texture: GLCM homogeneity, variance, contrast
- Damping ratio: backscatter inside vs. surrounding sea
- VH/VV ratio where dual-pol exists

**Morphology routes attribution:**
- Long thin streak, tapering → moving vessel → vessel path
- Irregular blob, fixed across passes → wreck/platform/seep → infrastructure path

**Physical limits. Do not fight these, state them:**
- SAR cannot measure oil thickness, volume, or type
- Revisit is 6–12 days per point; spills can appear and disperse between passes
- Outside the wind window, "no detection" ≠ "no oil"
- Drift error grows with time; beyond ~24 h backtracking, origin uncertainty is large

---

## PIPELINE

```
Sentinel-1 GRD (VV, + VH if present)
   │
   ├─ ingest/          calibrate Sigma0 dB, speckle filter, land mask, tile 512
   │
   ├─ detect/stage_b   U-Net (ResNet-34 encoder) → sea / oil / look-alike / ship / land
   │
   ├─ detect/lookalike PHYSICS CHECK  ← the differentiator
   │                   wind + shape + texture + damping + VH/VV → P(oil)
   │
   ├─ detect/morphology  linear → vessel path
   │                     blob   → infrastructure path
   │                     known seep → discard
   │
   ├─ drift/           OpenOil BACKWARD from polygon
   │                   CMEMS currents + ERA5 wind → origin + time + uncertainty
   │
   ├─ attribute/       AIS at origin ± window
   │                   score parity / proximity / temporality
   │                   check AIS-gap events → dark vessel
   │
   ├─ decision/        rank, bundle evidence, abstain if unclear
   │
   └─ api/ → ui/       map, drift animation, ranked suspects, caveats
```

`detect/stage_a` (a cheap tile screen that discards empty ocean before segmentation) is a **speed optimisation — build it last.**

---

## MODULE NOTES

**ingest/** — Calibrate GRD to Sigma0 dB. Speckle filter (Lee / refined Lee). Land mask from GSHHG. Tile 512×512 with overlap. **Cache tiles aggressively** — this step is slow and re-running it burns hours. Keep a *lightly* processed copy alongside the filtered one; heavy filtering destroys the texture the look-alike stage needs.

**detect/stage_b** — U-Net, ResNet-34 encoder (same choice as SkyTruth Cerulean, so it's defensible). Merge overlapping tile predictions by averaging confidence, then vectorise to polygons.

**detect/lookalike** — **Not a neural net.** A small gradient-boosted or logistic model over interpretable physical features, so every rejection can be explained on stage. Must print the reason: `"rejected: wind 1.2 m/s, below 3 m/s detection threshold"`.

**drift/** — OpenDrift's OpenOil, backward mode. Readers: `reader_copernicusmarine` (currents), `reader_netCDF_CF_generic` (ERA5 wind). Seed particles across the polygon, integrate backwards, output an origin probability field. **Disable weathering when running backwards** — those processes aren't reversible.

**attribute/** — Pull AIS for the scene bbox, 8 h before to 6 h after image time. Per candidate track:
- **parity** — how parallel the track is to the slick's long axis
- **proximity** — distance from slick head to nearest track point
- **temporality** — how recently the vessel passed

Combine, return top 3. Separately query AIS-gap events at the origin window: a vessel going dark exactly where a slick starts is a stronger signal than one that stayed visible.

**decision/** — Evidence bundle carries the wind value, the rejected look-alikes and why, drift uncertainty, and every sub-score. Abstain when top-two scores are within noise.

---

## DATA — ALL OPENLY DOWNLOADABLE

We have **no access to the Krestenitis dataset** (needs a supervisor's institutional email; unavailable to us). Everything below is open, no gatekeeping.

| Need | Source | How |
|---|---|---|
| SAR imagery | Copernicus Data Space Ecosystem | OData/STAC API, free account |
| SAR imagery (alt) | ASF DAAC | `asf_search` + NASA Earthdata login |
| SAR imagery (alt) | AWS Open Data | `s3://sentinel-s1-l1c --no-sign-request` |
| **Pixel segmentation** | Trujillo-Acatitla et al. 2024, Zenodo | Open. 2,850 patches, half oil / half look-alike-or-background, **pixel-wise masks**. This is the segmentation training set |
| **Look-alike analysis** | Yang et al. 2025, PANGAEA `10.1594/PANGAEA.980773` | **Metadata open; IMAGES ARE NOT.** The bulk archive answers **401 Unauthorized** and refers you to the PI, exactly like Krestenitis. What *is* open is the tabular export (`?format=textfile`, 2.4 MB): 5,515 annotated objects with class, acquisition time, patch and object corners in degrees, and the **source Sentinel-1 product id**. Those products are on the AWS mirror, so `scripts/extract_features.py` reads the ORIGINAL calibrated imagery instead of the gated JPGs — better than the dataset as distributed, since it yields real Sigma0 dB. **There are NO k-means cluster labels**: the only subgroup split published is `oc/ow/nc/nw` (oil/no-oil × coast/water). Boxes, not masks |
| **Cross-domain** | Deep-SAR SOS | GitHub `CUG-URS/CBDNet-main`, Kaggle mirror. 8,070 patches, ALOS PALSAR (Gulf of Mexico) + Sentinel-1 (Persian Gulf). Use as the *generalisation* test — different basins, different sensor |
| Ships / dark vessels | xView3-SAR | iuu.xview.us, free signup. **Take validation split only** — full set is hundreds of GB |
| Weak labels + validation | SkyTruth Cerulean API | `api.cerulean.skytruth.org` — open, no key. Historical slick polygons incl. Indian waters |
| Wind | ERA5 | Copernicus Climate Data Store, `cdsapi`, `~/.cdsapirc` |
| Currents | Copernicus Marine | `copernicusmarine` toolbox, `GLOBAL_ANALYSISFORECAST_PHY_001_024` |
| AIS (Indian waters) | Global Fishing Watch | Free non-commercial token. ~72 h delay. All vessel types, plus AIS-gap events |
| AIS (development) | Danish Maritime Authority | `http://web.ais.dk/aisdata/` — plain HTTP, no auth, ~2 GB/day |
| Land mask | GSHHG | Bundled with OpenDrift |

**The look-alike model is NOT fitted on Yang et al.** It was attempted and
abandoned: the dataset boxes oil objects but labels whole patches for
look-alikes, which makes `damping_db`, `elongation` and `compactness`
incomparable between the classes and turns `log_area` into leakage. The one
result that survived — look-alikes sit below the wind window 1.5x as often as
real oil — is in `docs/lookalike-fit-attempt.md`. Read it before trying again.

**Do not use Microsoft Planetary Computer** — Hub deprecated.

**Never bulk-download blindly.** Target working set ~50 GB.

Credentials go in `.env`. Never commit them.

---

## DATA SPLITS — KEEP SEPARATE

```
data/dev/            SOS + Zenodo + one Danish AIS day. Train and measure here.
data/demo_internal/  One clean linear bilge-dump slick + dense AIS. For 26 August.
data/demo_finale/    MSC ELSA 3 (Kerala, 24–27 May 2025, 09°18.75'N 076°08.16'E)
                     + one low-wind look-alike scene we correctly reject
                     + a persistent slick (Taylor Energy MC-20, 28.936N -88.970W)
                       as demo insurance — always shows a slick on any pass date
```

**Never evaluate on demo data.** Never train on it either.

---

## THE DEFAULT STACK — BUILD THIS FIRST

The bake-off exists to *challenge* this choice, not to make it from scratch. Build this, get numbers, then let the sweep try to beat it.

### The two decisions that matter more than the model

**1. Work at ~80 m resolution, not 10 m.** Sentinel-1 GRD IW arrives at ~10 m pixels. Oil slicks are hundreds of metres to kilometres across — none of that detail is signal, it's just speckle. Downsample by 8× before inference (this is exactly what Cerulean does). That is **64× fewer pixels**, which is a bigger speed win than any architecture choice, and it *improves* accuracy by suppressing speckle noise. Get this right and everything downstream is cheap.

**2. Cascade before you segment.** Most tiles in an ocean scene contain nothing. A cheap binary screen that discards them means the segmentation model runs on ~10% of the scene. Another order of magnitude, for almost no accuracy cost.

Together these two decisions are worth more than switching between any two backbones on this list. Do them first.

### Segmentation model — default

**U-Net with an ImageNet-pretrained ResNet-34 encoder.** Via `segmentation_models_pytorch`.

Reasons, in order:
- **It's what SkyTruth Cerulean uses.** When a jury asks why this architecture, "the operational open-source system for this exact task uses it" is a complete answer.
- **It behaves well on small labelled sets.** We have ~2,850 pixel-labelled patches. CNN inductive bias beats transformer flexibility in that regime.
- Skip connections preserve the thin, elongated structures that vessel discharges produce.
- ~24M params — fine-tunes fully on a consumer GPU, no head-only compromise needed.

**Primary challenger: SegFormer-B0** (~3.8M params). If it wins, it wins on both axes at once — fewer parameters, faster inference, and better global context, which matters here because whether a dark patch is oil depends on the state of the sea *around* it. Worth the sweep slot.

Everything else in the bake-off is there to map the frontier, not because it's likely to win.

### Training configuration — fix these, vary only the architecture

- Input: 512×512 tiles at ~80 m/px, 25% overlap
- Channels: VV, plus VH where present (handle single-pol by duplicating VV)
- Loss: **Dice + focal, equally weighted.** Oil is well under 1% of pixels; plain cross-entropy converges to "sea everywhere"
- Mixed precision (AMP) throughout, gradient accumulation instead of large batches
- Augmentation: flips, 90° rotations, mild brightness/contrast jitter. **No elastic or perspective warping** — it destroys the speckle statistics the model needs
- Encoder pretrained, never random init

**Keep wind out of the network.** It's tempting to add wind as a fourth input channel, and it would probably help slightly. Don't. Wind stays a separate, interpretable stage so we can print *"rejected: wind 1.2 m/s"* on stage. An explainable rejection is worth more to us than a fractional IoU gain buried in a tensor.

### Speed budget

Target: **a full Sentinel-1 GRD scene, end to end, in under two minutes** on the demo machine.

Optimisations in order of payoff:
1. 80 m resolution (64× fewer pixels)
2. Cascade screen (~10× fewer tiles reach segmentation)
3. FP16 inference (~2×)
4. Batched tiles (8–16)
5. Cached preprocessing — never re-tile a scene twice
6. ONNX Runtime / INT8 — **only if 1–5 don't get there.** Skip it if we're behind schedule; a slower model that works beats a fast one still being debugged at hour 33

Profile before optimising. One stage will dominate and the rest won't matter.

---

## MODEL SELECTION PROTOCOL

We do not pick a model by intuition. We run a **bake-off**: fine-tune several candidate architectures on a small sample, test on a disjoint sample, and pick by evidence.

Build this harness **early**, before optimising any single model. Choosing the wrong backbone and then tuning it for a week is the most expensive mistake available to us.

### Candidates to compare

At minimum:
- **U-Net + ResNet-34 (the default — everything else must beat it)**
- DeepLabv3+ (the published literature baseline)
- **SegFormer-B0 (primary challenger)** and SegFormer-B2
- U-Net + EfficientNet-B0 (efficiency-oriented)
- One deliberately tiny model (MobileNet-class) purely to map the accuracy/latency frontier

### Protocol — identical for every candidate

- **Few-shot fine-tune** at N = 50, 100, 200 training samples, then test on a **disjoint** held-out set. The sample-efficiency curve matters: the model that wins at N=200 is not always the one that wins at N=50, and we have limited labelled data.
- Freeze everything else. Same seed, same augmentation, same schedule, same loss (Dice/focal), same input size. **Only the architecture varies.**
- **Three seeds per configuration.** Report mean ± std. A single run difference of two points is noise, and picking on noise is worse than picking arbitrarily.
- Log every run to a results table: config, dataset version, seed, all metrics below.

### The decider — which slice actually separates models

**Overall mIoU is not the decider.** It is dominated by the sea and land classes, which every model scores in the 90s on. Two models can differ by 15 points on the thing we care about and by two points on mIoU.

Rank candidates on these, in this order:

1. **Look-alike false-positive rate, split coast vs water.** Yang et al. publish `nc`/`nw` — no-oil at the coast and on open water — and nothing finer. An earlier version of this file claimed twelve offshore and five coastal k-means subgroups; that is not in the released data, and any per-cluster claim built on it would be unsupportable. Report FP rate on each of the two subsets, and separately on the **low-wind slice** (ERA5 below the window), which we can construct ourselves because every patch carries a timestamp and position. **This is the primary decider.**
2. **Oil IoU on small slicks** — bottom quartile by area. Models diverge most on faint, small targets, and those are the ones that matter operationally (a huge obvious slick needs no AI).
3. **Cross-domain generalisation.** Train on Mediterranean data (Zenodo/PANGAEA), test on Deep-SAR SOS (Persian Gulf, Gulf of Mexico, different sensor). Our target is Indian waters, which appears in *no* training set — so the model that transfers best across basins is the model most likely to work where we actually need it. **Weight this heavily.**
4. **Sample efficiency** — the N=50 → N=200 curve. A model that reaches usable performance on 50 samples is worth more to us than one that needs 500.
5. **Calibration** — Expected Calibration Error and a reliability diagram. We abstain based on confidence, so a model whose 0.8 doesn't mean 0.8 breaks the abstention logic regardless of its IoU.

Explicitly **not** deciders: overall mIoU, pixel accuracy, sea-class IoU, land-class IoU. Compute them, don't rank on them.

### Efficiency metrics — measure, don't assume

Report for every candidate, measured on the actual demo hardware:

- **Parameter count** and checkpoint size on disk
- **Peak VRAM** during training and during inference (training VRAM decides what we can fine-tune locally)
- **Latency per 512×512 tile** at batch 1 and batch 8 — **median of ≥50 runs after 10 warm-up runs discarded**. Never report a mean; a single GC pause skews it.
- **Extrapolated full-scene time** = tiles per GRD scene × per-tile latency, plus preprocessing. This is the number that decides whether the demo feels alive.
- **Model load / cold-start time** — matters because the API loads once at startup.
- **CPU-only latency** — our fallback if the demo machine has no GPU.
- **Throughput** in tiles/second sustained.

### Selection rule

State the latency ceiling **before** running the bake-off, so the choice can't be rationalised afterwards: *full Sentinel-1 scene end-to-end in under X minutes on the demo machine.*

Then: **among candidates that meet the latency ceiling, pick the one with the lowest look-alike FP rate on the coast and low-wind subsets, tie-broken by cross-domain oil IoU.**

Produce a single comparison table — one row per model, columns for each decider metric plus each efficiency metric — and a scatter plot of look-alike FP rate against full-scene latency. That plot is a strong slide, and it shows a jury we chose by evidence rather than by fashion.

---

## PIPELINE EVALUATION

Separate from model selection. Once a backbone is chosen:

- **The wind ablation — our contribution.** False-positive rate with and without the wind stage, broken down **per look-alike cluster**. "Wind fusion cut false positives on the low-wind cluster by X%" is a far stronger claim than an aggregate number.
- **Attribution on MSC ELSA 3**: does our top-ranked candidate match the documented vessel? What is the drift-origin error against the known wreck position?
- **Agreement with Cerulean** on Indian-waters scenes — how often do we detect what an operational system detected?
- **Abstention behaviour**: what fraction goes to AMBER, and what is accuracy on the non-abstained remainder? Report the coverage/risk curve.
- **End-to-end latency** broken down by stage, so we know what to optimise.

**Baselines to beat:** a plain segmentation model with nearest-ship matching (what most competing teams will build), and the published detector baseline that ships with the Yang et al. dataset.

**Honesty note.** We do not have Krestenitis, so we cannot quote its published benchmark (DeepLabv3+ at 65.06% mIoU / 53.38% oil IoU / 55.40% look-alike IoU) as a comparable score. Cite it only as context for how hard the task is, clearly labelled as a different dataset and split. Never imply our numbers sit on the same scale.

---

## TRAPS

- **Class imbalance.** Oil is a fraction of a percent of pixels. Use **Dice or focal loss**. Plain cross-entropy converges to predicting "sea everywhere" and reports 99% accuracy while being useless.
- **Empty API responses.** CDSE, GFW and CMEMS return *empty results*, not errors, when a bbox is slightly wrong or no pass exists on that date. Assert non-empty. Print counts. Exit non-zero.
- **Lat/lon ordering** differs between GeoJSON, Shapely and most APIs. Get it wrong and everything lands silently in the wrong ocean.
- **VH is not always present.** Degrade the look-alike stage gracefully on single-pol scenes; never crash.
- **GRD scenes are ~1 GB.** Tile early, cache, never hold a full scene in memory.
- **Backward drift ≠ forward drift with a minus sign** for nonlinear processes. Disable weathering; note the limitation.
- **Don't extract texture features from the aggressively filtered image.** Use the lightly-processed copy.

---

## REPO LAYOUT

```
core/contracts.py     shared dataclasses — the interface between all modules
core/config.py        YAML config loading
ingest/               calibration, speckle, land mask, tiling, caching
detect/
  stage_a.py          tile screen (build last)
  stage_b.py          segmentation
  lookalike.py        physics-based rejection
  morphology.py       linear vs blob routing
drift/                OpenOil backward runs
attribute/
  ais.py              GFW client
  scoring.py          parity / proximity / temporality
  dark_vessel.py      AIS-gap detection
decision/             ranking, evidence bundle, abstention
api/                  FastAPI endpoints
scripts/              fetch_*.py, train.py, eval.py — run by teammates
data/                 dev/ demo_internal/ demo_finale/
models/               weights (gitignored)
tests/                mirrors module tree
```

---

## CONTRACTS

`core/contracts.py` is the interface between every module. Write it first. Freeze it. Flag any change to the team.

```python
@dataclass
class Scene:
    scene_id: str
    acquired_at: datetime
    bbox: tuple[float, float, float, float]   # minlon, minlat, maxlon, maxlat
    vv_path: Path
    vh_path: Path | None
    orbit_direction: Literal["ASCENDING", "DESCENDING"]

@dataclass
class WindContext:
    speed_ms: float
    direction_deg: float
    source: str                 # "ERA5" | "ASCAT"
    window_score: float         # 0-1, graded: peaks ~5-6 m/s, decays below ~3 and above ~10

@dataclass
class SlickCandidate:
    candidate_id: str
    scene_id: str
    polygon_wkt: str
    area_km2: float
    elongation: float
    compactness: float
    damping_ratio: float
    wind: WindContext
    p_oil: float                    # after look-alike rejection
    rejected_reason: str | None     # "wind 1.2 m/s below threshold"
    morphology: Literal["linear", "blob", "unknown"]

@dataclass
class DriftOrigin:
    lon: float
    lat: float
    estimated_at: datetime
    uncertainty_km: float
    n_particles: int

@dataclass
class VesselCandidate:
    mmsi: str
    name: str | None
    vessel_type: str | None
    flag: str | None
    parity: float        # 0-1
    proximity: float     # 0-1
    temporality: float   # 0-1
    score: float
    went_dark: bool      # AIS gap coincident with origin
    evidence: str        # human-readable

@dataclass
class Attribution:
    candidate_id: str
    origin: DriftOrigin | None
    source_type: Literal["vessel", "infrastructure", "natural_seep", "unknown"]
    candidates: list[VesselCandidate]   # ranked, may be empty
    abstained: bool
    abstain_reason: str | None
```

---

## STUBS ARE PERMANENT

Every module ships a stub returning valid fake output, selectable by config. If segmentation dies on demo day, its stub returns a plausible polygon and the rest still runs. **Never delete a stub to tidy up.**

---

## SCRIPTS TEAMMATES RUN

These are used by people who don't read Python. Config-driven. No code edits.

```bash
python scripts/fetch_sentinel.py --config configs/fetch_elsa3.yaml
python scripts/fetch_ais.py      --config configs/fetch_elsa3.yaml
python scripts/fetch_wind.py     --config configs/fetch_elsa3.yaml
python scripts/train.py          --config configs/train_baseline.yaml
python scripts/eval.py           --run runs/<run_name>
python scripts/bakeoff.py        --config configs/bakeoff.yaml
```

**Every fetch script prints what it actually got** — scene count, IDs, dates, bytes. Exit non-zero on empty.

`train.py` must: start from pretrained encoder weights (never scratch); support head-only training (fit a consumer GPU); use Dice/focal loss; write checkpoints and `results.json` in a fixed schema.

`eval.py` must print a table a non-technical person can paste into a spreadsheet.

`bakeoff.py` runs the full model-selection sweep: every candidate architecture × N ∈ {50,100,200} × 3 seeds, then emits one comparison table and the FP-rate-vs-latency scatter plot. It must be resumable — the sweep takes hours and will be interrupted.

---

## PACKAGING

GitHub from hour zero. Push after every working module. `README.md` with exact setup commands, a `Makefile`, working `configs/`.

Docker for training — `docker/Dockerfile.train`, CUDA base, pinned deps. Success condition on a fresh machine:

```bash
docker build -t oilspill-train -f docker/Dockerfile.train .
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/runs:/app/runs \
  oilspill-train --config configs/train_baseline.yaml
```

Plus `make train CONFIG=...` and a CPU fallback tag. **Test the image on someone else's machine before the internal round.**

---

## BUILD ORDER

1. `core/contracts.py` + all stubs — unblocks the team
2. `scripts/fetch_sentinel.py` — nothing happens without imagery
3. `ingest/` — calibration, tiling, caching
4. `detect/stage_b` — segmentation on SOS. Get *something* detecting first
5. `scripts/fetch_wind.py` + `detect/lookalike` — the differentiator
6. `detect/morphology`
7. `attribute/ais.py` + `attribute/scoring.py`
8. `drift/` — OpenOil backward
9. `attribute/dark_vessel.py`
10. `decision/` + `api/`
11. `detect/stage_a` — speed, last

**Segmentation before drift.** A working detector with a crude nearest-ship match is a demo. A perfect drift model with no detector is nothing.

---

## CODING STANDARDS

- Python 3.11, type hints on every function, no notebooks in the repo
- Tests before implementation; tests use real files from `data/dev/`, never in-memory fakes
- One module per Claude Code session — long sessions drift and rewrite working code
- Commit after every working module

---

## WHAT WE CLAIM

**Defensible:** near-real-time detection (3–24 h); measurable look-alike false-positive reduction from wind fusion; probabilistic ranked attribution; dark-vessel flagging via AIS gaps; validation against a real Indian incident.

**Must be softened:** "real-time" → "near-real-time". "Identifies the responsible vessel" → "ranks probable sources with confidence". Oil volume/thickness/type → SAR cannot measure these. "Continuous monitoring" → 6–12 day revisit gaps.

Every judge who knows remote sensing will test these. Saying them ourselves, first, is worth more than any accuracy number.
