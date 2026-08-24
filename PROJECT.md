# PROJECT.md — What We Are Building, in Plain Language

**Smart India Hackathon 2026 · Problem SIH26143 · Ministry: NTRO · Theme: Space Technology**

This document is for everyone on the team, whatever your background. No prior knowledge assumed. Read it once and you'll understand the whole project.

---

## 1. The problem

Ships produce oily waste — engine sludge, tank washings, bilge water. Disposing of it properly at port costs money and takes time. So some ships wait until they're far out at sea, usually at night, and pump it overboard. It takes twenty minutes. Nobody is watching.

This happens constantly, worldwide. Added up, these small routine dumps put more oil into the ocean than the famous tanker disasters do.

There are real accidents too. In May 2025 the container ship **MSC ELSA 3** sank off the Kerala coast carrying furnace oil and diesel, and the wreck kept leaking for weeks.

Either way, the same question, and nobody in India can currently answer it automatically:

> **There is oil in the water. Who put it there?**

## 2. Why this is hard

**You can't just look.** Ordinary satellite cameras are blocked by clouds, and half the time the ocean is in darkness.

So we use **radar satellites** instead. A radar satellite sends radio pulses down at the sea and listens for the echo. Choppy water scatters the signal in all directions, so the sea comes back **bright**. Oil flattens the tiny ripples on the surface, so an oiled patch bounces the signal away from the satellite and appears **dark**.

Radar sees through cloud, and works at night. That's why every serious oil-spill monitoring system uses it.

**Here's the catch, and it is the whole difficulty of this project:**

Lots of things flatten ripples. Very calm wind. Algae blooms. Rain cells. The wake behind a ship. Upwelling currents. All of them make dark patches that look exactly like oil on radar.

These are called **look-alikes**. Telling oil from look-alikes is where every existing system struggles. Published research models score in the 90s at recognising plain open sea — and only in the 50s and 60s at separating oil from look-alikes. That gap is where our project lives.

## 3. What already exists

Being honest about this matters, because judges will ask.

- **Europe** runs an operational service called CleanSeaNet that does roughly this for EU waters.
- **SkyTruth**, an American non-profit, runs an open-source system called **Cerulean** that detects slicks and suggests probable vessels. Their code is public. It is the closest thing to what we're building, and we will study it and measure ourselves against it.
- **India** has excellent satellite capability — ISRO imaged the Kerala spill within days — but no automated pipeline that connects a detected slick to a specific ship.
- **Dozens of student projects** have done "detect oil with a neural network, then find the nearest ship." That's the baseline we must beat, not the bar we aim for.

**So what makes ours different?** Four things, and we should be able to recite them:

1. We **reject look-alikes using physics**, not just a bigger neural network
2. We **drift the slick backwards in time** to find where it actually started
3. We give **ranked suspects with confidence scores**, not one accusation
4. We **detect ships that switched off their transponder** — the ones actually trying to hide

## 4. How the system works, step by step

### Step 1 — Get the picture
Download a radar image of a patch of ocean from Sentinel-1, a free European satellite. Clean it up: correct the brightness values, reduce the grainy noise radar always has, mask out land, and chop it into small tiles the computer can handle.

### Step 2 — Find the dark patches
A neural network looks at each tile and marks every pixel as one of five things: open sea, oil, look-alike, ship, or land. This gives us outlines of every suspicious dark shape.

### Step 3 — Is it actually oil?
**This is our main contribution.** For every dark shape found, we check:

- **What was the wind doing there?** We download the wind data for that exact place and time. Oil is only visible on radar in a specific wind range — roughly 3 to 10 metres per second. If the wind was nearly still, calm water alone explains the darkness. Reject it.
- **What shape is it?** Oil dumped by a moving ship forms a long thin streak. Algae forms irregular blobs.
- **What texture does it have?** Oil and algae have measurably different surface patterns.
- **How dark is it compared to the water around it?**
- **How does it respond to two different radar polarisations?** Oil and look-alikes respond differently. Free extra evidence.

Deliberately, this stage is **not** another neural network. It's a simple model over these physical measurements, so we can explain every single rejection out loud: *"rejected — wind was 1.2 metres per second, below the detection threshold."* A judge can follow that. Nobody can follow a neural network's hunch.

### Step 4 — What kind of source?
- **Long thin streak** → a moving ship was discharging → go to vessel attribution
- **Irregular blob that stays in one place** → a wreck, an oil platform, or a natural seep on the seabed → different path entirely

Natural seeps matter enormously. Oil leaks out of the seabed naturally in many places, and blaming a passing ship for a natural seep would be the worst mistake this system could make.

### Step 5 — Rewind the ocean
This is the part that makes the demo memorable.

The slick has been drifting since the moment it was dumped, pushed by currents and wind. So we take the ocean current data and wind data for the hours beforehand, and **run them backwards in time**. On screen, the oil crawls backwards across the map and settles onto a point.

That gives us a place **and** a time. Now the question becomes answerable.

### Step 6 — Who was there?
Ships broadcast their identity and position over a radio system called **AIS** — think of it as a transponder. We ask: which ships were at that spot, at that time, moving in a direction that matches the streak?

### Step 7 — Rank the suspects
Not one accusation. A ranked list, each with a confidence score, based on three measurements:

- **Parity** — how closely the ship's course lines up with the direction of the slick
- **Proximity** — how near it passed to where the slick begins
- **Temporality** — how recently it went by

### Step 8 — And if nobody was there?
Often the most interesting answer. A ship can switch its transponder off before dumping.

So we also look for **gaps** — a ship that was broadcasting normally, went silent right where the slick starts, then reappeared afterwards. A vessel going dark at exactly the moment oil appears is a *stronger* signal than one that stayed honest.

### Step 9 — Show the evidence
A map with the slick outline, the wind conditions, the backwards-drift animation, the ranked candidate vessels with their scores, and — stated plainly on screen — that this is a lead for investigators, not proof of guilt.

## 5. The honest limits

We say these ourselves, before any judge says them to us. Every competing team will claim real-time certainty, and anyone who knows remote sensing will know that's impossible.

| Limit | Reality |
|---|---|
| **Not real-time** | Satellite images reach us 3–24 hours after the pass. Free ship-tracking data is about 3 days delayed. |
| **Wind-dependent** | Oil is only reliably visible on radar in a middle band of wind — published studies put it between about 2–3 m/s at the low end and 7–12 m/s at the high end, with fuzzy edges. Too calm and everything looks like oil; too rough and the oil vanishes into the waves. |
| **Gappy coverage** | The satellite passes over any given point every 6–12 days. A dump can happen and disperse entirely in between. |
| **Can't measure amount** | Radar tells us oil is present and what shape it is. Not how much, not what type. |
| **Correlation isn't proof** | A ship being in the right place doesn't prove it dumped. We rank probabilities and say so. |

Being upfront about all five is a strength, not a weakness.

## 6. Where our data comes from

Everything is free. Nothing needs a supervisor, an institution, or a payment.

| What | Where |
|---|---|
| Radar satellite images | Copernicus Data Space (Europe), free account |
| Training examples of oil | Trujillo-Acatitla 2024 dataset on Zenodo — 2,850 images, open |
| Training examples of look-alikes | Yang et al. 2025 on PANGAEA — open, and sorted into 17 types of look-alike |
| Testing on a different ocean | Deep-SAR SOS — Gulf of Mexico and Persian Gulf, open on GitHub |
| Ship detections in radar | xView3 dataset, free signup |
| Reference detections to check ourselves against | SkyTruth Cerulean's open API |
| Wind data | ERA5, from the Copernicus Climate Data Store |
| Ocean currents | Copernicus Marine Service |
| Ship positions (Indian waters) | Global Fishing Watch — free for non-commercial use |
| Ship positions (for development) | Danish Maritime Authority — completely open, no login |

**A note on datasets.** The old standard benchmark for this task (Krestenitis, 2019) needs a request from a faculty supervisor's institutional email, which we don't have. It turns out not to matter — a newer dataset published in December 2025 by Yang and colleagues is completely open, and is better suited to what we're doing.

Here's why it's better. Ordinary datasets just say "this is oil" or "this isn't." Yang's team went further: they took all the *non-oil* dark patches and sorted them into 17 groups by what caused them — calm wind, internal waves, algae, rain, and so on. That means we can say exactly which kind of false alarm our system fixes and which it still falls for. Nobody using the old dataset can do that. It's the single most useful thing we could have been handed, and it's free.

We can still mention the old benchmark's published scores to show how hard this problem is — we just can't claim our numbers sit on the same scale, because they're measured on different data.

## 6a. The two decisions that make it fast

Before any of the clever machine learning, two plain engineering choices do most of the work.

**We deliberately blur the picture.** The satellite gives us images at about 10 metres per pixel. Oil slicks are hundreds of metres to kilometres across, so all that fine detail is just noise — literally, radar images are grainy by nature. We shrink the image by a factor of eight before analysing it. That's 64 times fewer pixels to process, and it actually makes detection *better* by removing the grain. This single choice speeds us up more than any change of model could.

**We skip the empty ocean.** Most of any satellite image is featureless water. A quick, cheap check throws those parts away first, so the expensive analysis only runs on the roughly one tenth of the image that has anything in it.

Together these two are worth more than any amount of model tuning. Target: a full satellite scene processed end to end in under two minutes on a normal laptop.

## 6b. How we choose which model to use

There are a dozen reasonable neural network designs for this task. Rather than picking one because it sounds impressive, we test them.

**The bake-off.** We take each candidate design, train it on a small number of examples, test it on a different set it has never seen, and compare. Same settings for all of them, so the only thing that differs is the design itself. We run each one three times with different random starts, because a single run can look better or worse purely by luck.

**What we judge them on — and this part matters.** The obvious score is "what percentage of pixels did it get right." That number is useless here, because most of any ocean picture is just plain sea, and every model gets plain sea right. Two models can look nearly identical on that score while being wildly different at the only thing we care about.

So we rank models on:

1. **How often they mistake a look-alike for oil** — measured separately for each of the 17 look-alike types. Calm wind and internal waves are the ones that break real systems, so those count most.
2. **How well they find small, faint slicks.** A huge obvious spill needs no AI. The hard ones are where models differ.
3. **How well they work on an ocean they've never seen.** We train on Mediterranean data and test on the Gulf of Mexico. Our actual target is Indian waters, which appears in no training set at all — so whichever model travels best is the one most likely to work where we need it.
4. **How little data they need.** A model that works well on 50 examples is worth more to us than one needing 500, because labelled examples are scarce.
5. **Whether their confidence is honest.** Our system says "I'm not sure" when confidence is low. If a model says it's 80% sure but is only right 60% of the time, that whole feature breaks.

**We also measure speed**, on the actual laptop we'll demo from: how long one image tile takes, how long a full satellite scene takes, how much memory it needs, and how slow it is with no graphics card at all as a fallback.

**The rule we commit to in advance:** set a maximum acceptable time per scene *before* running any tests. Then among the models fast enough, pick whichever makes the fewest look-alike mistakes. Deciding the rule first stops us rationalising a favourite afterwards.

The output is one table and one chart — mistakes plotted against speed. That chart goes in the presentation, because it shows we chose by evidence.

## 7. Our three demo sets

Kept strictly separate. We never test on data we demo with, and never demo on data we trained with.

**Development set** — the open training datasets plus one day of Danish ship-tracking data. This is what we build and measure against. Never shown on stage.

**College internal round (26 August)** — one clean scene showing a classic bilge dump: a long dark streak with a ship track running parallel to it. Instantly readable on a projector. Runs start to finish in minutes.

**National finale** — three scenes:
1. **MSC ELSA 3, Kerala, May 2025.** Indian waters, and the responsible vessel is publicly known — so we can check our system's answer against the documented truth in front of the judges. This is the most persuasive thing we can possibly show.
2. **A calm-wind look-alike** that a naive model calls oil and our wind check correctly rejects. Demonstrates honesty and shows off our differentiator.
3. **A permanent slick** — there are places, like Taylor Energy in the Gulf of Mexico, where oil leaks continuously and shows up on nearly every satellite pass. This is our insurance policy: if we run something live on stage, it cannot come back empty.

## 8. Who does what

**Kunsh — all the code.** Every module: image processing, the detection model, the look-alike checks, the drift simulation, ship matching, scoring, and the backend. Also writes the download and training scripts that everyone else runs.

**Data (2 people).** Register the accounts. Run the download scripts. Collect the demo scenes. Check what comes back is actually correct — these APIs return *nothing* rather than an error when a coordinate is slightly wrong, so someone has to look at what actually arrived. Keep an inventory of what's in each folder.

**Training and evaluation (1 person).** Run the training script with different settings. Keep a results spreadsheet, one row per run. Report two numbers daily: our best score on data we didn't train on, and whether the gap between training and testing scores is growing. Use free Kaggle GPU hours when the local machine is busy.

**Design (2 people).** Figma first, then build. Three screens: a map showing detected slicks, a detail view with the drift animation and ranked suspects, and a system dashboard. Also owns the demo rehearsal and the backup video.

**Everyone.** Learn the five honest limits in section 5 well enough to say them out loud. Judges ask.

## 9. The demo, as the judges will see it

1. A radar image of the ocean appears. Mostly grey noise. A dark streak is visible if you know to look.
2. The system marks every dark patch — including several that aren't oil.
3. The wind data loads. Three patches are rejected on screen, each with its reason printed: *"wind 1.4 m/s — below detection threshold."* One survives.
4. The surviving slick is classified as a linear discharge from a moving vessel.
5. The drift animation runs backwards. The oil crawls back across the map to a point, with an uncertainty circle around it.
6. Ship tracks appear. Three candidates, ranked, with confidence scores and the reasoning behind each.
7. **Then the reveal:** for the Kerala scene, we show the documented answer. Our top candidate matches.
8. We finish by stating what the system cannot do.

That last step wins more points than it costs. Judges have sat through a dozen teams claiming perfection.

## 10. Glossary

- **SAR** — Synthetic Aperture Radar. The kind of satellite imaging that works through cloud and at night.
- **Sentinel-1** — the free European radar satellite we use.
- **GRD** — the specific Sentinel-1 image product we download.
- **AIS** — Automatic Identification System. The radio transponder ships broadcast their identity and position on.
- **MMSI** — a ship's unique AIS identity number.
- **Look-alike** — anything that makes a dark patch on radar but isn't oil.
- **Slick** — a patch of oil on the sea surface.
- **Bilge dump** — a ship illegally pumping oily waste overboard.
- **Dark vessel** — a ship that has switched off its AIS transponder.
- **Drift model** — a simulation of how ocean currents and wind move things around the sea.
- **Segmentation** — a neural network labelling every individual pixel of an image.
- **IoU** — Intersection over Union. The standard score for how well a predicted outline matches the true one.
- **Abstain** — the system deliberately returning "I don't know" instead of a guess.
- **Bake-off** — testing several model designs head-to-head under identical conditions to pick the best one.
- **Latency** — how long the system takes to produce an answer.
- **Calibration** — whether a model's stated confidence matches how often it's actually right.
