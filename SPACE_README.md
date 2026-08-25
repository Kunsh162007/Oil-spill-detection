---
title: Oil Spill Detection and Vessel Attribution
emoji: 🛰️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: SAR oil-spill detection with physics-based look-alike rejection and ranked vessel attribution
---

# Oil Spill Detection & Vessel Attribution

**SIH 2026 · SIH26143 · NTRO · Space Technology**

Detects oil slicks in Sentinel-1 radar imagery, rejects natural look-alikes
using physics rather than a bigger network, drifts each slick backwards
through ocean currents to estimate where it started, and returns a **ranked
list of candidate vessels** — never a single accusation.

Source: https://github.com/Kunsh162007/Oil-spill-detection

## What you can do here

- **Overview** — system state, why each look-alike was rejected, per-stage latency
- **Active** — slicks in imagery fresh enough (< 72 h) to forecast a present position
- **Past incidents** — historical detections, 3,414 documented spills worldwide, and every rejection with its reason

Click a slick for the timeline: **Origin → Observed → Now**, the physics
breakdown, and ranked vessel routes with departure and arrival ports.

## Honest limits, stated up front

1. **Not real-time.** Imagery arrives 3–24 h after acquisition; free AIS lags ~72 h.
2. **Wind-dependent.** Oil is reliably visible on radar only between roughly 2–3 and 7–12 m/s.
3. **Gappy coverage.** Revisit is 6–12 days; a spill can appear and disperse between passes.
4. **No quantity.** SAR cannot measure oil thickness, volume or type.
5. **Correlation is not proof.** Ranked candidates are investigative leads.

Attribution abstains wherever AIS coverage is absent — that is a missing
input, not a confident answer.
