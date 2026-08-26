# Fitting the look-alike model on Yang et al. (PANGAEA) — a negative result

**Date:** 26 August 2026
**Dataset:** Yang et al. 2025, PANGAEA `10.1594/PANGAEA.980773`
**Outcome:** the model was **not** refitted. The hand-set physical priors remain
in use. One genuine finding came out of the attempt and is recorded below.

This exists so the attempt is not repeated blindly, and so the one usable
result stays quotable with its caveats attached.

---

## Why we tried

`CLAUDE.md` names *"look-alike false-positive rate"* as the **primary decider**
for model selection, and the look-alike stage ships hand-set logistic weights
rather than fitted ones. Fitting them on a published, independently-labelled
dataset would turn a defensible *process* claim into a defensible *accuracy*
claim.

## What the dataset actually is

Three assumptions in `CLAUDE.md` did not survive contact with the record. All
three have since been corrected there.

| Claim | Reality |
|---|---|
| "Open, CC-BY 4.0, no request" | The **images are gated**. `download.pangaea.de/dataset/980773/allfiles.zip` returns **401 Unauthorized** — *"you need to be authorized… contact the principal investigator"*. Exactly the Krestenitis situation this dataset was chosen to avoid. |
| "K-means clustered into 12 offshore + 5 coastal subgroups" | **No cluster labels exist** in the release. The only subgroup split published is `oc/ow/nc/nw` — oil/no-oil × coast/water. |
| The patches are the useful part | The **tabular metadata** is open and is the more useful half: 5,515 annotated objects with class, timestamp, patch and object corners in degrees, and the **source Sentinel-1 product id**. |

That last row is a genuine win. Those products are on the AWS Open Data mirror,
unauthenticated, so `scripts/extract_features.py` reads the **original
calibrated imagery** instead of the gated 8-bit JPGs. Features come out as real
Sigma0 in dB — better than the dataset as distributed.

## What was extracted

40 source products, sampled class-balanced from the 1,181 available, up to 10
objects each: **174 objects, 73 oil / 101 look-alike, all dual-pol**, with real
ERA5 wind per object from the Open-Meteo archive.

One bug worth remembering: `scripts/fetch_aws.py:read_window` returns
"dB **or** DN", and for these GRDs it is raw digital numbers. A first pass
assumed decibels and produced damping between −101 and +112 dB against a
physical range of roughly 0–15, with the sign inverted. The extractor now routes
arrays through `ingest/calibrate.py:to_sigma0_db`, the same path the pipeline
uses.

## Why the fit was abandoned

The blocker is **annotation convention**, not code.

| | oil | look-alike |
|---|---|---|
| median area | **1.1 km²** | **153.2 km²** |
| what is labelled | a tight box around the slick | the entire 640×640 patch |

Yang annotates *objects* for oil and labels *whole patches* for look-alikes —
reasonably, since there is nothing to box when the whole scene is the
phenomenon. But it makes the classes geometrically incomparable, and that
poisons most of our feature set.

Class separation, measured as |median difference| / pooled σ:

| feature | separation | verdict |
|---|---|---|
| `log_area` | **2.03** | **leakage.** Strongest signal in the table, and it encodes annotation style, not physics. A model fitted here learns "small = oil". |
| `texture_contrast` | 1.33 | suspect. A tight box straddles dark oil *and* bright sea, so oil shows **higher** contrast (45.8 vs 15.3) — that is the box edge. |
| `texture_homogeneity` | 1.06 | suspect, same reason |
| `wind_window_score` | 0.91 | **trustworthy** — see below |
| `wind_speed_ms` | 0.72 | trustworthy |
| `vh_vv_ratio` | 0.30 | weak |
| `compactness` | 0.14 | no information — both classes are rectangles |
| `elongation` | 0.08 | no information — look-alikes are always the same square |
| `damping_db` | **0.00** | **no signal.** For a whole-patch label the "surrounding ring" falls outside the patch, so damping is ~0 by construction. |

A logistic model fitted on this would score well on hold-out and perform
**worse in production**, because the leakage does not exist at inference time.
`models/lookalike.json` was deliberately not written.

## The result that is real

Wind is derived from geolocation and timestamp, not from window geometry, so
the annotation asymmetry cannot touch it. It is the one feature computed
identically here and at inference.

```
oil         n=73    wind median 4.24 m/s    21% below the detection window
look-alike  n=101   wind median 2.97 m/s    32% below the detection window
```

**Look-alikes occur below the wind window at roughly 1.5× the rate of real
oil**, across 174 independently-labelled objects.

This is direct empirical support for the premise the whole differentiator rests
on: the wind gate rejects the right population. It is quotable because it
depends on no part of our detector — only on published labels and ERA5.

Caveats that must travel with it: 174 objects from 40 of 1,181 products; the
coast subset is thin (19 objects); Eastern Mediterranean, 2019 only.

## What would make a real fit possible

1. **Ask the PI for the XML annotations.** They contain look-alike *object*
   boxes, which would make the classes geometrically comparable and restore
   `damping_db`, `elongation` and `compactness`. This is the clean fix.
2. **Label a look-alike subset ourselves** with comparable boxes. Defensible,
   but then the labels are ours and must be described that way.

Until either happens the look-alike stage uses hand-set physical priors, and
says so at runtime:

```
No fitted look-alike model at models/lookalike.joblib - using hand-set
physical priors. Run scripts/train.py --stage lookalike to fit real weights.
```

## Reproducing

```bash
curl -s "https://doi.pangaea.de/10.1594/PANGAEA.980773?format=textfile" \
    -o data/raw/pangaea/pang.txt
python scripts/extract_features.py --max-scenes 40 --max-objects-per-scene 10
# inspect data/dev/lookalike_features.csv before fitting anything on it
```
