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
.venv/Scripts/python.exe -m pytest              # 151 tests
.venv/Scripts/python.exe scripts/eval.py --latency
.venv/Scripts/python.exe scripts/eval.py --wind-ablation
```

---

## What the map does

| Action | Result |
|---|---|
| Load the page | World map of every confirmed slick across all analysed scenes |
| Click a slick | Physics breakdown, drift origin, ranked candidate vessels |
| Press play | Backward-drift animation — the oil crawls back to its origin, with the date and time on each frame |
| Hover a vessel track | Full voyage: departure port and time, arrival port and time, distance, speed, declared AIS destination |
| A dashed track segment | The vessel's AIS went silent across that leg |

Vessel routes are drawn as the whole observed passage with **A** (track start)
and **B** (track end) pins, plus a dotted projection of where the course points
next. The panel states plainly that those endpoints are the limits of the AIS
window queried, not of the vessel's entire voyage.

---

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
- Full pipeline end to end, 1.2 s per synthetic scene (budget: 120 s)
- Physics rejection of calm wind, storm wind, weak damping, blooms, rain cells, internal waves
- Backward drift, verified to 0.07 km against hand-calculated displacement
- Ranked attribution with dark-vessel detection
- Abstention on every uncertainty case in `CLAUDE.md` rule 5
- 151 tests, 80% coverage
- Live Sentinel-1 search against CDSE (verified: 4 dual-pol scenes over the MSC ELSA 3 wreck site)

**Not yet done — needs data or credentials:**

| Gap | What unblocks it |
|---|---|
| **No trained segmentation model.** Runs a classical adaptive dark-patch detector instead, and says so in `/api/health`. | Download Zenodo (2,850 patches) + Deep-SAR SOS, then `scripts/train.py` |
| **Look-alike weights are hand-set priors, not fitted.** Announced loudly at startup. | Yang et al. PANGAEA dataset → `scripts/train.py --stage lookalike` |
| **Wind and currents are synthetic** in the demo config. Every drift origin is tagged `analytical-advection-SYNTHETIC`. | `scripts/fetch_wind.py` (ERA5) and CMEMS credentials |
| **GPU unavailable on this machine** — only 2.5 GB free disk, CUDA wheel needs ~6 GB. Runs on CPU. | Free ~8 GB, then reinstall torch from the cu128 index |
| **OpenDrift not installed** (conda-first). Analytical RK4 integrator used instead. | `pip install opendrift`, or accept the fallback |

Nothing above is faked. Each degraded path announces itself in logs, in
`/api/health`, and on screen.

---

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
