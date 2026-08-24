# Oil Spill Detection & Vessel Attribution

**SIH 2026 · Problem SIH26143 · NTRO · Space Technology**

Find oil slicks in Sentinel-1 radar imagery, reject the natural look-alikes
using physics, drift the slick backwards through ocean currents to estimate
where it started, and return a **ranked list of candidate vessels** — never a
single accusation.

Read `PROJECT.md` for the plain-language explanation. `CLAUDE.md` is the
engineering contract.

---

## Quick start

```bash
# 1. Environment (Python 3.11 — PyTorch and rasterio do not support 3.14)
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -r requirements-ml.txt

# 2. Generate the synthetic demo scenes (no download, no credentials)
.venv/Scripts/python.exe scripts/make_demo_scene.py
.venv/Scripts/python.exe scripts/make_demo_scene.py --calm-wind \
    --name CALM_WIND_DEMO --bbox "74.20,9.05,74.80,9.65" --seed 21

# 3. Run the map
.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
```

Open **http://127.0.0.1:8000**.

On Linux/macOS use `.venv/bin/python` and `make setup && make demo && make serve`.
(`make` is not installed by default on Windows; the commands above are the
same targets spelled out.)

### Verify it works

```bash
.venv/Scripts/python.exe -m pytest              # 214 tests
.venv/Scripts/python.exe scripts/eval.py --latency
.venv/Scripts/python.exe scripts/eval.py --wind-ablation
```

---

## What the map does

Three tabs over one world map.

| Tab | Shows |
|---|---|
| **Overview** | System state, why look-alikes were rejected, latency per stage |
| **Active** | Slicks detected in imagery fresh enough (< 72 h) to forecast a present position |
| **Past incidents** | Our historical detections, the documented-spill registry, and every rejected look-alike with its reason |

Clicking a slick opens the full attribution:

| Element | What it gives you |
|---|---|
| **Timeline** | `Origin → Observed → Now`. Click any point to fly the map there. Origin comes from backward drift, Observed is the satellite record, Now is forward drift to the present. |
| **Is this actually oil?** | Confidence tier with the evidence behind it |
| **Physics check** | Wind, damping, morphology and per-feature log-odds weights |
| **Ranked vessels** | Parity / proximity / temporality, plus the full voyage |
| **Vessel routes** | The whole passage with **A** (departure) and **B** (arrival) pins, ports, times, and the AIS-declared destination. Dashed segments are AIS gaps. |
| **Drift animation** | Play the slick back to its origin, timestamped frame by frame |

### Two kinds of dot, never blended

- **Orange / brown** — something *our pipeline detected* in SAR imagery
- **Pink** — a *documented incident* somebody recorded, independent of any model

Conflating a detection with a confirmed event is the error this project exists
to avoid, so the two are separate layers with separate colours.

### Only actual oil is presented as oil

Every detection is graded into a tier, and only the top two appear as oil:

| Tier | Meaning |
|---|---|
| **Confirmed** | Physics says oil AND a public incident registry records a spill here at this time |
| **Probable** | Strong physical evidence: good wind, clear damping, discharge-like shape |
| **Possible** | Consistent with oil but weakly supported — held back by default |
| **Insufficient** | Does not meet the bar; kept with its reason so the rejection can be audited |

Corroboration is checked against **3,414 documented spills** worldwide. It is
weighted heavily but capped, so it can never carry a candidate on its own — and
a physics rejection is never overturned by it.

## Pipeline

```
Sentinel-1 GRD (VV, +VH)
  ├─ ingest/       calibrate to Sigma0 dB, refined-Lee speckle, land mask, tile 512 @ ~80 m
  ├─ detect/stage_a  cheap tile screen — discards ~80% of empty ocean
  ├─ detect/stage_b  segmentation (U-Net/ResNet-34; classical detector until trained)
  ├─ detect/lookalike  PHYSICS CHECK ← the differentiator
  ├─ detect/wavetrain  scene-level internal-wave rejection
  ├─ detect/morphology linear → vessel path; blob/known seep → excluded
  ├─ drift/        OpenOil (or analytical RK4) BACKWARD to origin + uncertainty
  ├─ attribute/    AIS tracks, parity/proximity/temporality, AIS-gap detection
  ├─ decision/     rank, bundle evidence, abstain when unclear
  └─ api/ → ui/    map, drift animation, ranked suspects, caveats
```

### The look-alike stage

Not a neural network. Hard physical gates, then a small logistic model over
interpretable features, so every rejection prints its reason:

```
rejected: wind 1.2 m/s is below the 2.0 m/s floor — calm water is
          indistinguishable from oil on SAR
rejected: one of 9 near-parallel bands (axis ~0°) evenly spaced every 3.3 km
          — the signature of an internal-wave train, not a vessel discharge
```

---

## Current state — what is real and what is not

**Working and verified:**

- Full pipeline end to end on **real Sentinel-1 imagery**
- **56 real detections** across 21 genuine Sentinel-1 scenes (Gulf of Mexico, 2018–2020), many corroborated against the Taylor Energy MC-20 site
- **Fine-tuned U-Net/ResNet-34** on real oil-spill data — Oil IoU **0.54** (see below)
- **3,414 documented spills** worldwide from NOAA IncidentNews plus a curated world catalogue
- Physics rejection of calm wind, storm wind, weak damping, blooms, rain cells and internal-wave trains
- Backward drift verified to 0.07 km, forward drift to 0.02 km, against hand-calculated displacements
- Ranked attribution with dark-vessel detection and full voyage reconstruction
- Abstention on every uncertainty case in `CLAUDE.md` rule 5
- **214 tests**, 80% coverage
- Live Sentinel-1 search against CDSE (verified over the MSC ELSA 3 wreck site)

### Training

Fine-tuned on Zenodo record 4672426 (CC-BY-4.0): 21 real Sentinel-1 scenes of
Gulf of Mexico spills with binary pixel masks, cross-referenced against NOAA
incident reports.

```bash
python scripts/fetch_incidents.py      # documented spill registry
python scripts/prepare_dataset.py      # 28,993 patches, read on demand
python scripts/train.py --config configs/train_gom.yaml
python scripts/register_scenes.py      # put the real scenes on the map
```

Patches are read from their parent GeoTIFFs on demand rather than materialised
(5.7 GB saved). The prep step verifies label quality automatically — the
coordinate convention in that dataset is `(column, row)` marking the patch
*centre*; reading it as `(row, col)` top-left drops usable labels from 73% to
21%, so it is checked rather than assumed.

Selection is on **oil IoU, never mean IoU** — mIoU is dominated by the sea
class, which every model scores in the 90s on.

**Not yet done — needs data or credentials:**

| Gap | What unblocks it |
|---|---|
| **Look-alike weights are hand-set priors**, not fitted. Announced loudly at startup. | Yang et al. PANGAEA dataset → `scripts/train.py --stage lookalike` |
| **Wind and currents are synthetic or climatological** in the demo configs. Every drift origin is tagged accordingly. | `scripts/fetch_wind.py` (ERA5) and CMEMS credentials |
| **No active detections** — all imagery on disk is from 2018–2025, so everything is filed as a past incident. | `scripts/fetch_sentinel.py` for imagery from the last 72 h |
| **GPU unavailable** on this machine (disk-constrained); trained on CPU. | Free ~6 GB, reinstall torch from the cu128 index |
| **OpenDrift not installed** (conda-first). Analytical RK4 integrator used instead. | `pip install opendrift`, or accept the fallback |
| **Cerulean API now requires a key** — CLAUDE.md assumed it was open; it returns 403 as of this build. | Request access, or rely on the NOAA registry (already integrated) |

Nothing above is faked. Each degraded path announces itself in logs, in
`/api/health`, and on screen.

## Scripts teammates run

```bash
python scripts/fetch_sentinel.py --config configs/fetch_elsa3.yaml [--list-only]
python scripts/fetch_wind.py     --config configs/fetch_elsa3.yaml
python scripts/fetch_ais.py      --config configs/fetch_elsa3.yaml --source gfw
python scripts/train.py          --config configs/train_baseline.yaml
python scripts/eval.py           --run runs/<name> | --wind-ablation | --latency
python scripts/bakeoff.py        --config configs/bakeoff.yaml [--latency-only]
```

Every fetch script prints what it actually got and **exits non-zero on empty**.
CDSE, GFW and CMEMS return empty results rather than errors when a bbox is
slightly wrong, and a silent empty download looks exactly like "nothing was
there".

---

## Credentials

Copy `.env.example` to `.env`. All free, none needed for the synthetic demo.

| Service | Needed for |
|---|---|
| Copernicus Data Space | Sentinel-1 imagery |
| Global Fishing Watch | AIS in Indian waters |
| Copernicus Marine | ocean currents |
| CDS (`~/.cdsapirc`) | ERA5 wind |

---

## The five honest limits

1. **Not real-time.** Imagery arrives 3–24 h after acquisition; free AIS lags ~72 h.
2. **Wind-dependent.** Oil is reliably visible only between roughly 2–3 and 7–12 m/s.
3. **Gappy coverage.** Revisit is 6–12 days; a dump can appear and disperse between passes.
4. **No quantity.** SAR cannot measure oil thickness, volume or type.
5. **Correlation is not proof.** Ranked candidates are investigative leads.

Every one of these is stated in the API payloads and on screen.
