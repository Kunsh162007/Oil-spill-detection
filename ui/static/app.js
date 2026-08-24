/* Oil Spill Detection & Vessel Attribution - map console.
 *
 * Two views:
 *   world  - every confirmed slick on one map; click one to inspect it
 *   detail - one slick: physics, backward-drift origin, ranked vessels
 *
 * Every surface that names a vessel also shows the disclaimer. Correlation
 * is not evidence, and the UI must never let that caveat fall off.
 */

const COLORS = {
  oil: "#ff5c39",
  rejected: "#6b7a90",
  abstain: "#f0b232",
  origin: "#35d0ba",
  vessel: "#4c8dff",
  darkVessel: "#c65cff",
};

const state = {
  map: null,
  worldLayer: null,
  detailLayer: null,
  slicks: [],
  view: "world",
  current: null,
  anim: { frames: [], idx: 0, timer: null, marker: null, trail: null },
};

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n) ? "--" : Number(n).toFixed(d));

function fmtTime(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  return d.toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", timeZone: "UTC",
  }) + " UTC";
}

/* ---------------------------------------------------------------- map --- */

function initMap() {
  state.map = L.map("map", { zoomControl: true, worldCopyJump: true }).setView([12, 74], 5);

  // Dark basemap so the slick colours carry the meaning, not the terrain.
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; OpenStreetMap &copy; CARTO | Sentinel-1 imagery: Copernicus',
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(state.map);

  state.worldLayer = L.layerGroup().addTo(state.map);
  state.detailLayer = L.layerGroup().addTo(state.map);
}

function clearDetail() {
  stopAnimation();
  state.detailLayer.clearLayers();
}

/* -------------------------------------------------------------- world --- */

async function loadWorld() {
  try {
    const [slicks, health] = await Promise.all([
      fetch("/api/slicks").then((r) => r.json()),
      fetch("/api/health").then((r) => r.json()),
    ]);
    renderWorld(slicks, health);
  } catch (err) {
    $("panel-body").innerHTML =
      `<div class="empty">Could not reach the API.<br><br>${err}</div>`;
  }
}

function renderWorld(data, status) {
  state.slicks = data.features || [];
  state.view = "world";
  clearDetail();
  state.worldLayer.clearLayers();

  if (status && status.timeliness) $("timeliness").textContent = status.timeliness;

  const bounds = [];
  state.slicks.forEach((f) => {
    const layer = drawSlick(f, state.worldLayer);
    if (layer && layer.getBounds) bounds.push(layer.getBounds());
  });

  if (bounds.length) {
    state.map.fitBounds(bounds.reduce((a, b) => a.extend(b)), { padding: [70, 70], maxZoom: 9 });
  }

  $("panel-title").textContent = "Detected slicks";
  $("panel-sub").textContent =
    `${data.meta.n_slicks} confirmed across ${data.meta.n_scenes} scene(s). Click a slick for attribution.`;

  renderWorldList(data);
}

function drawSlick(feature, layerGroup) {
  const p = feature.properties;
  const g = feature.geometry;
  const color = p.is_oil ? (p.abstained ? COLORS.abstain : COLORS.oil) : COLORS.rejected;

  let layer;
  if (g.type === "Polygon" && g.coordinates[0].length >= 4) {
    const latlngs = g.coordinates[0].map(([lon, lat]) => [lat, lon]);
    layer = L.polygon(latlngs, {
      color, weight: 2, fillColor: color,
      fillOpacity: p.is_oil ? 0.34 : 0.12,
      dashArray: p.is_oil ? null : "4,4",
    });
  } else {
    const [lon, lat] = g.coordinates;
    layer = L.circleMarker([lat, lon], {
      radius: 7, color, fillColor: color, fillOpacity: 0.6, weight: 2,
    });
  }

  layer.bindTooltip(
    `<b>${p.candidate_id}</b><br>${fmt(p.area_km2)} km&sup2; &middot; P(oil) ${fmt(p.p_oil)}` +
    (p.is_oil ? "" : `<br><i>${p.rejected_reason || "rejected"}</i>`),
    { direction: "top" }
  );
  layer.on("click", () => { if (p.is_oil) openDetail(p.candidate_id); });
  layer.addTo(layerGroup);
  return layer;
}

function renderWorldList(data) {
  const body = $("panel-body");
  if (!state.slicks.length) {
    body.innerHTML = `<div class="empty">
      No confirmed slicks.<br><br>
      Generate a demo scene:<br>
      <code>python scripts/make_demo_scene.py</code>
    </div>`;
    return;
  }

  const synthetic = state.slicks.some(
    (f) => (f.properties.wind || {}).source === "synthetic"
  );

  body.innerHTML =
    (synthetic
      ? `<div class="synthetic-warn"><b>Synthetic environmental data.</b>
         Wind and currents are simulated for this demo, so drift origins are a
         demonstration of the method, not a measurement.</div>`
      : "") +
    state.slicks.map(slickCardHTML).join("") +
    `<div class="disclaimer">${data.meta.disclaimer}</div>`;

  body.querySelectorAll("[data-cid]").forEach((el) => {
    el.addEventListener("click", () => openDetail(el.dataset.cid));
  });
}

function slickCardHTML(f) {
  const p = f.properties;
  const pill = p.abstained
    ? '<span class="pill abstain">Insufficient evidence</span>'
    : '<span class="pill oil">Oil</span>';
  const top = p.top_candidate;

  return `<div class="slick-card" data-cid="${p.candidate_id}">
    <div class="card-top">
      <span class="card-id">${p.candidate_id}</span>${pill}
    </div>
    <div class="card-metrics">
      <span><b>${fmt(p.area_km2)}</b> km&sup2;</span>
      <span>P(oil) <b>${fmt(p.p_oil)}</b></span>
      <span>wind <b>${fmt(p.wind.speed_ms, 1)}</b> m/s</span>
    </div>
    ${top ? `<div class="card-suspect">
        Top candidate: <b>${top.name || top.mmsi}</b> &middot; ${fmt(top.score)}
        ${top.went_dark ? '<span class="pill dark">went dark</span>' : ""}
      </div>` : ""}
    ${p.abstained ? `<div class="card-reason">${p.abstain_reason}</div>` : ""}
  </div>`;
}

/* ------------------------------------------------------------- detail --- */

async function openDetail(candidateId) {
  state.view = "detail";
  $("panel-body").innerHTML = '<div class="spinner">Backtracking drift&hellip;</div>';

  let detail, trace;
  try {
    detail = await fetch(`/api/slicks/${candidateId}`).then((r) => r.json());
    const res = await fetch(`/api/slicks/${candidateId}/backtrace`);
    trace = res.ok ? await res.json() : null;
  } catch (err) {
    $("panel-body").innerHTML = `<div class="empty">Could not load ${candidateId}<br><br>${err}</div>`;
    return;
  }

  state.current = { detail, trace };
  renderDetailPanel(detail, trace);
  drawDetailMap(detail, trace);
}

function renderDetailPanel(d, trace) {
  const wind = d.wind || {};
  const origin = d.origin;

  $("panel-title").textContent = d.candidate_id;
  $("panel-sub").textContent = d.abstained
    ? "Insufficient evidence to rank a source"
    : `${d.vessels.length} candidate vessel(s), ranked by correlation`;

  const abstain = d.abstained
    ? `<div class="section"><h3>Abstention</h3>
        <div class="card-reason" style="border-left-color:var(--abstain)">
          ${d.abstain_reason}</div></div>`
    : "";

  const originBlock = origin
    ? `<div class="section">
        <h3>Estimated origin &mdash; backward drift</h3>
        <dl class="kv">
          <dt>Position</dt><dd>${fmt(origin.lat, 4)}&deg;N, ${fmt(origin.lon, 4)}&deg;E</dd>
          <dt>Released at</dt><dd>${fmtTime(origin.estimated_at)}</dd>
          <dt>Uncertainty</dt><dd>&plusmn;${fmt(origin.uncertainty_km, 1)} km</dd>
          <dt>Backtracked</dt><dd>${fmt(origin.backtrack_hours, 1)} h</dd>
          <dt>Particles</dt><dd>${origin.n_particles}</dd>
          <dt>Method</dt><dd style="font-size:10px">${origin.method}</dd>
        </dl>
        ${trace ? playbarHTML(trace) : ""}
        ${!origin.reliable
          ? `<div class="card-reason" style="border-left-color:var(--abstain);margin-top:9px">
             Beyond ~24 h of backtracking the origin is a wide blur, not a point.</div>`
          : ""}
      </div>`
    : "";

  $("panel-body").innerHTML = `
    <button class="back-link" id="back-btn">&larr; All slicks</button>
    ${abstain}
    <div class="section">
      <h3>Physics check</h3>
      <dl class="kv">
        <dt>P(oil)</dt><dd>${fmt(d.slick.p_oil)}</dd>
        <dt>Area</dt><dd>${fmt(d.slick.area_km2)} km&sup2;</dd>
        <dt>Morphology</dt><dd>${d.slick.morphology}</dd>
        <dt>Wind speed</dt><dd>${fmt(wind.speed_ms, 1)} m/s</dd>
        <dt>Wind window</dt><dd>${fmt(wind.window_score)}</dd>
        <dt>Wind source</dt><dd>${wind.source}</dd>
      </dl>
      ${contributionsHTML(d.evidence.physics_contributions)}
    </div>
    ${originBlock}
    <div class="section">
      <h3>Candidate vessels &mdash; ranked, not accused</h3>
      ${d.vessels.length
        ? d.vessels.map(vesselHTML).join("")
        : '<div class="empty">No vessel tracks near the estimated origin.</div>'}
    </div>
    <div class="disclaimer">${d.disclaimer}</div>
  `;

  $("back-btn").addEventListener("click", showWorld);
  $("panel-body").querySelectorAll("[data-mmsi]").forEach((el) => {
    el.addEventListener("click", () => focusVessel(el.dataset.mmsi));
  });
  if (trace) wirePlaybar(trace);
}

function contributionsHTML(contrib) {
  if (!contrib || !Object.keys(contrib).length) return "";
  const rows = Object.entries(contrib)
    .filter(([k]) => k !== "_bias")
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const max = Math.max(...rows.map(([, v]) => Math.abs(v)), 0.001);

  return `<div style="margin-top:11px"><h3>Evidence weights (log-odds)</h3>
    <div class="contrib">${rows.map(([k, v]) => {
      const w = (Math.abs(v) / max) * 50;
      return `<div class="contrib-row">
        <span style="color:var(--muted)">${k.replace(/_/g, " ")}</span>
        <div class="contrib-bar"><div class="contrib-axis"></div>
          <div class="contrib-fill ${v >= 0 ? "pos" : "neg"}" style="width:${w}%"></div>
        </div>
        <span class="bar-val">${v >= 0 ? "+" : ""}${fmt(v)}</span>
      </div>`;
    }).join("")}</div></div>`;
}

function vesselHTML(v) {
  return `<div class="vessel rank-${v.rank}" data-mmsi="${v.mmsi}">
    <div class="vessel-head">
      <div>
        <div class="vessel-name">#${v.rank} ${v.name || "Unknown vessel"}
          ${v.went_dark ? '<span class="pill dark">went dark</span>' : ""}</div>
        <div class="vessel-mmsi">MMSI ${v.mmsi}${v.vessel_type ? " &middot; " + v.vessel_type : ""}${v.flag ? " &middot; " + v.flag : ""}</div>
      </div>
      <div class="vessel-score">${fmt(v.score)}</div>
    </div>
    <div class="bars">
      ${barHTML("parity", v.parity)}
      ${barHTML("proximity", v.proximity)}
      ${barHTML("temporality", v.temporality)}
    </div>
    ${voyageHTML(v.voyage)}
    <div class="evidence-text">${v.evidence}</div>
  </div>`;
}

function voyageHTML(vy) {
  if (!vy) return "";
  const pa = vy.projected_arrival;
  return `<div class="voyage">
    <div class="voyage-leg">
      <span class="pin start">A</span>
      <div>
        <div class="voyage-place">${vy.from.nearest_port || "open sea"}</div>
        <div class="voyage-time">${fmtTime(vy.from.at)}</div>
      </div>
    </div>
    <div class="voyage-line">
      <span>${fmt(vy.distance_km, 0)} km &middot; ${fmt(vy.duration_hours, 1)} h &middot; ${fmt(vy.mean_speed_knots, 1)} kn</span>
    </div>
    <div class="voyage-leg">
      <span class="pin end">B</span>
      <div>
        <div class="voyage-place">${vy.to.nearest_port || "open sea"}</div>
        <div class="voyage-time">${fmtTime(vy.to.at)}</div>
      </div>
    </div>
    ${vy.declared_destination
      ? `<div class="voyage-meta">Declared destination (AIS): <b>${vy.declared_destination}</b></div>`
      : ""}
    ${pa && pa.port
      ? `<div class="voyage-meta">Course points toward <b>${pa.port}</b>, ~${pa.hours} h on &mdash; projection, not a declared route</div>`
      : ""}
    <div class="voyage-note">${vy.coverage_note}</div>
  </div>`;
}

function barHTML(label, value) {
  return `<div class="bar-row">
    <span class="bar-label">${label}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${(value || 0) * 100}%"></div></div>
    <span class="bar-val">${fmt(value)}</span>
  </div>`;
}

/* ------------------------------------------------------ detail on map --- */

function drawDetailMap(d, trace) {
  clearDetail();
  state.worldLayer.clearLayers();
  const bounds = [];

  // 1. The observed slick.
  if (d.slick.polygon && d.slick.polygon.length >= 4) {
    const latlngs = d.slick.polygon.map(([lon, lat]) => [lat, lon]);
    const poly = L.polygon(latlngs, {
      color: COLORS.oil, weight: 2, fillColor: COLORS.oil, fillOpacity: 0.3,
    }).addTo(state.detailLayer);
    poly.bindTooltip("Observed slick", { direction: "top" });
    bounds.push(poly.getBounds());
  }

  // 2. Backward drift path, and the uncertainty circle at the origin.
  if (d.origin) {
    const track = (d.origin.track || []).map((p) => [p.lat, p.lon]);
    if (track.length > 1) {
      L.polyline(track, {
        color: COLORS.origin, weight: 2.5, opacity: 0.85, dashArray: "6,5",
      }).addTo(state.detailLayer).bindTooltip("Backward drift path", { sticky: true });
    }

    L.circle([d.origin.lat, d.origin.lon], {
      radius: d.origin.uncertainty_km * 1000,
      color: COLORS.origin, weight: 1, opacity: 0.5,
      fillColor: COLORS.origin, fillOpacity: 0.09,
    }).addTo(state.detailLayer)
      .bindTooltip(`Origin uncertainty ±${fmt(d.origin.uncertainty_km, 1)} km`, { sticky: true });

    const originMarker = L.circleMarker([d.origin.lat, d.origin.lon], {
      radius: 8, color: COLORS.origin, fillColor: COLORS.origin,
      fillOpacity: 0.95, weight: 2,
    }).addTo(state.detailLayer);
    originMarker.bindTooltip(
      `<b>Estimated origin</b><br>${fmtTime(d.origin.estimated_at)}<br>±${fmt(d.origin.uncertainty_km, 1)} km`,
      { direction: "top", className: "origin-label" }
    ).openTooltip();
    bounds.push(L.latLngBounds([[d.origin.lat, d.origin.lon]]));
  }

  // 3. Full vessel routes: the whole observed passage, with where it began,
  //    where it ended, and where the course points next.
  d.vessels.forEach((v) => {
    if (!v.track || v.track.length < 2) return;
    const pts = v.track.map((p) => [p.lat, p.lon]);
    const color = v.went_dark ? COLORS.darkVessel : COLORS.vessel;
    const primary = v.rank === 1;

    // An AIS gap is a hole in the track. Drawn as a dashed connector so the
    // silence is visible rather than smoothed over into a continuous line.
    const segments = splitOnGaps(v.track);
    segments.forEach((seg) => {
      const line = L.polyline(seg.map((p) => [p.lat, p.lon]), {
        color, weight: primary ? 3.5 : 2, opacity: primary ? 0.95 : 0.55,
      }).addTo(state.detailLayer);
      line.bindTooltip(vesselTooltip(v), { sticky: true });
      line._mmsi = v.mmsi;
      bounds.push(line.getBounds());
    });
    for (let i = 0; i < segments.length - 1; i++) {
      const a = segments[i][segments[i].length - 1];
      const b = segments[i + 1][0];
      L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {
        color, weight: 2, opacity: 0.9, dashArray: "3,7",
      }).addTo(state.detailLayer)
        .bindTooltip("AIS silent across this leg", { sticky: true });
    }

    const vy = v.voyage;
    const first = v.track[0];
    const last = v.track[v.track.length - 1];

    addEndpoint(first, "start", color, primary,
      `<b>${v.name || v.mmsi}</b><br>Track begins${vy && vy.from.nearest_port ? " near " + vy.from.nearest_port : ""}<br>${fmtTime(first.at)}`);
    addEndpoint(last, "end", color, primary,
      `<b>${v.name || v.mmsi}</b><br>Track ends${vy && vy.to.nearest_port ? " near " + vy.to.nearest_port : ""}<br>${fmtTime(last.at)}` +
      (vy && vy.declared_destination ? `<br>Declared destination: ${vy.declared_destination}` : ""));

    // Where the course points after our AIS window runs out. Dotted, and
    // labelled a projection, because it is not an observation.
    if (primary && vy && vy.projected_arrival && vy.projected_arrival.lon != null) {
      const pa = vy.projected_arrival;
      L.polyline([[last.lat, last.lon], [pa.lat, pa.lon]], {
        color, weight: 1.5, opacity: 0.5, dashArray: "2,8",
      }).addTo(state.detailLayer)
        .bindTooltip(
          `Projected onward course${pa.port ? " toward " + pa.port : ""}` +
          `${pa.hours ? " (~" + pa.hours + " h)" : ""}<br><i>${pa.basis}</i>`,
          { sticky: true }
        );
    }
  });

  function addEndpoint(point, kind, color, primary, html) {
    const marker = L.marker([point.lat, point.lon], {
      icon: L.divIcon({
        className: "",
        html: `<div class="route-pin ${kind}" style="--pin:${color}">${kind === "start" ? "A" : "B"}</div>`,
        iconSize: [18, 18], iconAnchor: [9, 9],
      }),
      opacity: primary ? 1 : 0.75,
    }).addTo(state.detailLayer);
    marker.bindTooltip(html, { direction: "top" });
    bounds.push(L.latLngBounds([[point.lat, point.lon]]));
  }

  function vesselTooltip(v) {
    const vy = v.voyage;
    let html = `<b>#${v.rank} ${v.name || v.mmsi}</b><br>score ${fmt(v.score)}`;
    if (vy) {
      html += `<br>${vy.from.nearest_port || "open sea"} &rarr; ${vy.to.nearest_port || "open sea"}`;
      html += `<br>${fmt(vy.distance_km, 0)} km over ${fmt(vy.duration_hours, 1)} h at ${fmt(vy.mean_speed_knots, 1)} kn`;
    }
    html += `<br>closest ${fmt(v.closest_approach_km, 1)} km at ${fmtTime(v.closest_approach_at)}`;
    if (v.went_dark) html += "<br><b>AIS gap at the origin</b>";
    return html;
  }

  if (bounds.length) {
    state.map.fitBounds(bounds.reduce((a, b) => a.extend(b)), { padding: [80, 80] });
  }

  if (trace) setupAnimation(trace);
}

function splitOnGaps(track, gapMinutes = 45) {
  /* Break a track at AIS silences so a gap is drawn as a gap, not a
     straight line implying the vessel was tracked the whole way. */
  const segments = [];
  let current = [track[0]];
  for (let i = 1; i < track.length; i++) {
    const dt = (new Date(track[i].at) - new Date(track[i - 1].at)) / 60000;
    if (dt > gapMinutes) {
      segments.push(current);
      current = [];
    }
    current.push(track[i]);
  }
  if (current.length) segments.push(current);
  return segments.filter((s) => s.length > 1);
}

function focusVessel(mmsi) {
  document.querySelectorAll(".vessel").forEach((el) =>
    el.classList.toggle("active", el.dataset.mmsi === mmsi)
  );
  state.detailLayer.eachLayer((layer) => {
    if (layer._mmsi === mmsi && layer.getBounds) {
      state.map.fitBounds(layer.getBounds(), { padding: [90, 90] });
      layer.setStyle({ weight: 5, opacity: 1 });
    } else if (layer._mmsi) {
      layer.setStyle({ weight: 2, opacity: 0.4 });
    }
  });
}

function showWorld() {
  clearDetail();
  state.current = null;
  loadWorld();
}

/* ---------------------------------------------------------- animation --- */

function playbarHTML(trace) {
  return `<div class="playbar">
    <button class="playbtn" id="play-btn" title="Play the backward drift">&#9654;</button>
    <input type="range" class="scrub" id="scrub"
           min="0" max="${Math.max(trace.n_frames - 1, 0)}" value="0" />
    <span class="frame-time" id="frame-time">--</span>
  </div>`;
}

function setupAnimation(trace) {
  // Frames arrive oldest first, so playing forward shows the oil travelling
  // the way it actually did: from the estimated origin to where we saw it.
  state.anim.frames = trace.frames || [];
  state.anim.idx = 0;
  if (!state.anim.frames.length) return;

  const first = state.anim.frames[0];
  state.anim.marker = L.circleMarker([first.lat, first.lon], {
    radius: 7, color: "#fff", fillColor: COLORS.oil, fillOpacity: 0.95, weight: 2,
  }).addTo(state.detailLayer);
  state.anim.trail = L.polyline([[first.lat, first.lon]], {
    color: COLORS.oil, weight: 3, opacity: 0.75,
  }).addTo(state.detailLayer);

  showFrame(0);
}

function wirePlaybar(trace) {
  const btn = $("play-btn");
  const scrub = $("scrub");
  if (!btn || !scrub) return;

  btn.addEventListener("click", () => {
    if (state.anim.timer) { stopAnimation(); btn.innerHTML = "&#9654;"; }
    else { startAnimation(); btn.innerHTML = "&#10074;&#10074;"; }
  });
  scrub.addEventListener("input", (e) => {
    stopAnimation();
    btn.innerHTML = "&#9654;";
    showFrame(Number(e.target.value));
  });
}

function showFrame(i) {
  const frames = state.anim.frames;
  if (!frames.length) return;
  const idx = Math.max(0, Math.min(i, frames.length - 1));
  state.anim.idx = idx;
  const f = frames[idx];

  if (state.anim.marker) state.anim.marker.setLatLng([f.lat, f.lon]);
  if (state.anim.trail) {
    state.anim.trail.setLatLngs(frames.slice(0, idx + 1).map((p) => [p.lat, p.lon]));
  }

  const label = $("frame-time");
  if (label) {
    const hrs = f.hours_before;
    label.textContent =
      `${fmtTime(f.at)}${hrs > 0 ? ` (-${fmt(hrs, 1)} h)` : " (observed)"}`;
  }
  const scrub = $("scrub");
  if (scrub && Number(scrub.value) !== idx) scrub.value = idx;
}

function startAnimation() {
  stopAnimation();
  if (state.anim.idx >= state.anim.frames.length - 1) state.anim.idx = 0;
  state.anim.timer = setInterval(() => {
    if (state.anim.idx >= state.anim.frames.length - 1) {
      stopAnimation();
      const btn = $("play-btn");
      if (btn) btn.innerHTML = "&#9654;";
      return;
    }
    showFrame(state.anim.idx + 1);
  }, 220);
}

function stopAnimation() {
  if (state.anim.timer) {
    clearInterval(state.anim.timer);
    state.anim.timer = null;
  }
}

/* ---------------------------------------------------------------- boot -- */

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadWorld();
});
