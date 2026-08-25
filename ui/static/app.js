/* Oil Spill Detection & Vessel Attribution — map console.
 *
 * Three tabs over one map:
 *   overview   system state, what was rejected and why, latency
 *   slicks     our SAR detections; click one for full attribution
 *   documented confirmed spills from public incident registries
 *
 * The two kinds of dot on the map mean different things and are never
 * blended: an orange slick is something OUR pipeline detected, a pink dot is
 * an incident somebody recorded. Conflating a detection with a confirmed
 * event is exactly the error this project exists to avoid.
 */

const C = {
  oil: "#ff5b32", rejected: "#5f7488", abstain: "#f5a524",
  origin: "#2dd4bf", vessel: "#4c8dff", dark: "#c084fc",
  confirmed: "#34d399", incident: "#f472b6", past: "#9a6b5a",
};

const state = {
  map: null,
  layers: {},
  slicks: [],
  incidents: [],
  stats: null,
  health: null,
  tab: "overview",
  current: null,
  timeline: null,
  tlMarker: null,
  tlCircle: null,
  pastFilter: "detections",
  anim: { frames: [], idx: 0, timer: null, marker: null, trail: null },
};

if (typeof window !== "undefined") window.__app = state;

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) =>
  n === null || n === undefined || Number.isNaN(n) ? "--" : Number(n).toFixed(d);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtTime(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--";
  return d.toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", timeZone: "UTC",
  }) + " UTC";
}
function fmtDate(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "--";
  return d.toLocaleDateString("en-GB", {
    day: "2-digit", month: "short", year: "numeric", timeZone: "UTC",
  });
}

/* ---------------------------------------------------------------- map --- */

function initMap() {
  state.map = L.map("map", {
    zoomControl: true, worldCopyJump: true, minZoom: 2,
  }).setView([14, 60], 3);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution:
      '&copy; OpenStreetMap &copy; CARTO | Sentinel-1: Copernicus | Incidents: NOAA',
    subdomains: "abcd", maxZoom: 18,
  }).addTo(state.map);

  state.layers.slicks = L.layerGroup().addTo(state.map);
  state.layers.unattributed = L.layerGroup().addTo(state.map);
  state.layers.past = L.layerGroup().addTo(state.map);
  state.layers.rejected = L.layerGroup();
  state.layers.incidents = L.layerGroup().addTo(state.map);
  state.layers.detail = L.layerGroup().addTo(state.map);

  state.map.on("zoomend", () => {
    applyZoomStyling();
    scaleIncidentMarkers();
  });

  bindLayerToggle("layer-slicks", "slicks");
  bindLayerToggle("layer-unattributed", "unattributed");
  bindLayerToggle("layer-past", "past");
  bindLayerToggle("layer-rejected", "rejected");
  bindLayerToggle("layer-incidents", "incidents");
}

function bindLayerToggle(inputId, layerName) {
  const input = $(inputId);
  if (!input) return;
  input.addEventListener("change", () => {
    const layer = state.layers[layerName];
    if (input.checked) layer.addTo(state.map);
    else state.map.removeLayer(layer);
  });
}

function clearDetail() {
  stopAnimation();
  state.layers.detail.clearLayers();
}

/* --------------------------------------------------------------- boot --- */

async function boot() {
  try {
    const [health, slicks, incidents, stats] = await Promise.all([
      fetch(apiUrl("/api/health")).then((r) => r.json()),
      fetch(apiUrl("/api/slicks")).then((r) => r.json()),
      fetch(apiUrl("/api/incidents?limit=4000")).then((r) => (r.ok ? r.json() : { features: [], meta: {} })),
      fetch(apiUrl("/api/stats")).then((r) => (r.ok ? r.json() : null)),
    ]);

    state.health = health;
    state.slicks = slicks.features || [];
    state.incidents = incidents.features || [];
    state.stats = stats;
    state.slicksMeta = slicks.meta || {};
    state.incidentsMeta = incidents.meta || {};

    renderChips();
    drawSlickLayers();
    drawIncidentLayer();
    updateCounts();
    fitToData();
    renderTab("overview");
  } catch (err) {
    $("panel-body").innerHTML =
      `<div class="empty">Could not reach the API.<br><br>${esc(err)}</div>`;
  }
}

function renderChips() {
  const h = state.health || {};
  const seg = (h.backends || {}).segmentation || "unknown";
  const trained = !/classical/i.test(seg);
  const chip = $("chip-backend");
  chip.textContent = trained ? "trained U-Net" : "classical detector";
  chip.className = trained ? "chip" : "chip warn";
  chip.title = seg;
  $("chip-timeliness").textContent = h.timeliness || "near-real-time";
}

// A detection with no ranked vessel is still a real detection - the physics
// stage confirmed it as oil. What it lacks is an answer to "who". Listing it
// under "active detections" alongside attributed ones implies a lead that does
// not exist, so it gets its own bucket. CLAUDE.md rule 5: abstain when
// uncertain, and say so where the user can see it.
function activeSlicks() {
  return state.slicks.filter(
    (f) => f.properties.is_oil && f.properties.activity === "active" &&
           !f.properties.abstained);
}

function unattributedSlicks() {
  return state.slicks.filter(
    (f) => f.properties.is_oil && f.properties.activity === "active" &&
           f.properties.abstained);
}

function pastSlicks() {
  return state.slicks.filter(
    (f) => f.properties.is_oil && f.properties.activity !== "active");
}

function updateCounts() {
  const active = activeSlicks();
  $("count-slicks").textContent = active.length;
  $("count-unattributed").textContent = unattributedSlicks().length;
  $("count-past").textContent = pastSlicks().length;
  $("count-rejected").textContent =
    state.slicks.filter((f) => !f.properties.is_oil).length;
  $("count-incidents").textContent = state.incidents.length;
  const badge = $("badge-active");
  if (badge) badge.textContent = active.length;
}

function fitToData() {
  /* Default to a world view. The whole point of the incident layer is global
     coverage, so opening zoomed into one scene would hide it. */
  const pts = [];
  state.incidents.forEach((f) =>
    pts.push([f.geometry.coordinates[1], f.geometry.coordinates[0]]));
  state.slicks.forEach((f) => {
    if (f.properties.is_oil) pts.push(centroidOf(f));
  });
  if (pts.length < 2) {
    state.map.setView([20, 20], 2);
    return;
  }
  try {
    state.map.fitBounds(L.latLngBounds(pts), { padding: [40, 40], maxZoom: 4 });
  } catch (err) {
    state.map.setView([20, 20], 2);
  }
}

function centroidOf(feature) {
  const p = feature.properties;
  if (p.centroid) return [p.centroid[1], p.centroid[0]];
  const g = feature.geometry;
  if (g.type === "Point") return [g.coordinates[1], g.coordinates[0]];
  const ring = g.coordinates[0];
  const lon = ring.reduce((a, c) => a + c[0], 0) / ring.length;
  const lat = ring.reduce((a, c) => a + c[1], 0) / ring.length;
  return [lat, lon];
}

/* ------------------------------------------------------------- layers --- */

function drawSlickLayers() {
  state.layers.slicks.clearLayers();
  state.layers.unattributed.clearLayers();
  state.layers.past.clearLayers();
  state.layers.rejected.clearLayers();

  state.slicks.forEach((f) => {
    const p = f.properties;
    const isPast = p.activity === "historical";
    // A past detection is real but not current; it must not sit on the map
    // in the same colour as something we believe is out there right now.
    const target = !p.is_oil ? state.layers.rejected
      : isPast ? state.layers.past
      : p.abstained ? state.layers.unattributed : state.layers.slicks;
    const color = !p.is_oil ? C.rejected
      : isPast ? C.past : (p.abstained ? C.abstain : C.oil);
    const g = f.geometry;

    const [clat, clon] = centroidOf(f);
    const hasPolygon =
      g.type === "Polygon" && g.coordinates[0] && g.coordinates[0].length >= 4;

    // Two representations of the same slick. A 10 km slick is well under one
    // pixel at world zoom, so its polygon simply vanishes; the marker keeps it
    // findable. Above POLYGON_ZOOM the real outline is worth showing.
    const marker = L.circleMarker([clat, clon], {
      radius: markerRadius(p, state.map.getZoom()),
      color, fillColor: color,
      fillOpacity: !p.is_oil ? 0.35 : isPast ? 0.6 : 0.85,
      weight: isPast ? 1.5 : 2,
      className: p.is_oil && !isPast && !p.abstained
        ? "slick-marker live" : "slick-marker",
    });
    marker._slickProps = p;
    target.addLayer(marker);

    let layer = marker;
    if (hasPolygon) {
      const poly = L.polygon(g.coordinates[0].map(([lon, lat]) => [lat, lon]), {
        color, weight: isPast ? 1.5 : 2, fillColor: color,
        fillOpacity: !p.is_oil ? 0.12 : isPast ? 0.2 : 0.35,
        dashArray: p.synthetic ? "7,5" : (p.is_oil ? null : "4,4"),
      });
      poly._isSlickPolygon = true;
      target.addLayer(poly);
      layer = poly;
      if (p.is_oil) poly.on("click", () => openDetail(p.candidate_id));
      poly.bindTooltip(slickTooltip(p, isPast), { direction: "top" });
    }

    marker.bindTooltip(slickTooltip(p, isPast), { direction: "top" });
    if (p.is_oil) marker.on("click", () => openDetail(p.candidate_id));
  });

  applyZoomStyling();
}

function slickTooltip(p, isPast) {
  return `<b>${esc(p.label || p.candidate_id)}</b><br>` +
    `${fmt(p.area_km2)} km&sup2; &middot; P(oil) ${fmt(p.p_oil)}<br>` +
    `wind ${fmt((p.wind || {}).speed_ms, 1)} m/s` +
    (p.synthetic ? '<br><b class="synth-flag">SYNTHETIC DEMONSTRATION SCENE</b>' +
       '<br><i>fabricated imagery and invented AIS &mdash; not a real detection</i>' : "") +
    (isPast ? `<br><i>past incident &mdash; ${fmt(p.age_days, 0)} days old</i>` : "") +
    (p.corroborated
      ? `<br><b style="color:${C.confirmed}">confirmed by incident registry</b>` : "") +
    (p.is_oil ? "" : `<br><i>${esc(p.rejected_reason || "rejected")}</i>`);
}

// Below this zoom a slick polygon is smaller than a pixel, so the marker
// carries it instead.
const POLYGON_ZOOM = 8;

function markerRadius(p, zoom) {
  /* Grow markers as the view widens. Zoomed out the map is an index of where
     to look, so the dot must be findable; zoomed in the polygon takes over
     and the dot shrinks out of the way. */
  const base = zoom <= 3 ? 6 : zoom <= 5 ? 5.5 : zoom <= 7 ? 5 : 3.5;
  const emphasis =
    p.is_oil && p.activity === "active" && !p.abstained ? 1.35 : 1.0;
  return base * emphasis;
}

function applyZoomStyling() {
  const zoom = state.map.getZoom();
  const showPolygons = zoom >= POLYGON_ZOOM;
  [state.layers.slicks, state.layers.unattributed, state.layers.past,
   state.layers.rejected].forEach((group) => {
    group.eachLayer((layer) => {
      if (layer._isSlickPolygon) {
        layer.setStyle({ opacity: showPolygons ? 1 : 0, fillOpacity: showPolygons ? 0.3 : 0 });
      } else if (layer._slickProps) {
        layer.setRadius(markerRadius(layer._slickProps, zoom));
        layer.setStyle({ opacity: showPolygons ? 0.55 : 1 });
      }
    });
  });
}

function drawIncidentLayer() {
  state.layers.incidents.clearLayers();

  state.incidents.forEach((f) => {
    const p = f.properties;
    const [lon, lat] = f.geometry.coordinates;
    // Scale by spilled volume where known, so the map reads at a glance.
    const v = p.volume_m3 || 0;
    const radius = v > 0 ? Math.max(3.5, Math.min(13, 3.5 + Math.log10(v + 1) * 2.1)) : 3.5;
    const color = p.natural_seep ? "#a3a3a3" : C.incident;

    const marker = L.circleMarker([lat, lon], {
      radius: radius * incidentZoomScale(state.map.getZoom()),
      color, fillColor: color,
      fillOpacity: p.persistent ? 0.75 : 0.45,
      weight: p.persistent ? 2 : 1,
      className: "incident-marker",
    });
    marker._baseRadius = radius;
    marker
      .bindTooltip(
        `<b>${esc(p.name)}</b><br>` +
        `${esc(p.location || "")}<br>` +
        `${p.persistent ? "persistent source" : fmtDate(p.occurred_at)}` +
        (p.commodity ? `<br>${esc(p.commodity)}` : "") +
        (p.volume_m3 ? `<br>${Math.round(p.volume_m3).toLocaleString()} m&sup3;` : "") +
        `<br><i style="color:${C.confirmed}">documented incident &mdash; ${esc(p.source)}</i>`,
        { direction: "top" }
      )
      .addTo(state.layers.incidents);
  });
  scaleIncidentMarkers();
}

function incidentZoomScale(zoom) {
  /* At world zoom thousands of incidents overlap into a blob; shrinking them
     keeps the pattern readable, and they grow back as the view narrows. */
  return zoom <= 2 ? 0.62 : zoom <= 4 ? 0.85 : zoom <= 6 ? 1.0 : 1.25;
}

function scaleIncidentMarkers() {
  const scale = incidentZoomScale(state.map.getZoom());
  state.layers.incidents.eachLayer((layer) => {
    if (layer._baseRadius) layer.setRadius(layer._baseRadius * scale);
  });
}

/* --------------------------------------------------------------- tabs --- */

function renderTab(name) {
  state.tab = name;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  clearDetail();
  state.current = null;
  state.timeline = null;
  if (state.fromDetail) {
    fitToData();
    state.fromDetail = false;
  }

  if (name === "overview") renderOverview();
  else if (name === "active") renderActive();
  else if (name === "past") renderPast();
}


// What the analyses actually used, not what the service default config says.
// The health endpoint can only report the fallback config, so it announced
// "synthetic" for both while every real scene ran on ERA5 and CMEMS. Falls
// back to the health value only when the index carries no counts.
function fieldSummary(key, fallback) {
  const counts = (state.slicksMeta || {})[key] || {};
  // Collapse per-scene filenames to the product that produced them. Currents
  // are stored one file per scene, so the raw keys are
  // "cmems:S1C_IW_GRDH_1SDV_2026...nc" and the panel showed a truncated
  // filename where a source name belongs.
  const byProduct = {};
  for (const [name, n] of Object.entries(counts)) {
    const product = String(name).split(":")[0];
    byProduct[product] = (byProduct[product] || 0) + n;
  }
  const entries = Object.entries(byProduct).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return fallback || "--";
  if (entries.length === 1) return entries[0][0];
  // More than one means a fallback was used somewhere; show the split so a
  // partly-synthetic run can never look uniform.
  return entries.map(([name, n]) => `${name} (${n})`).join(", ");
}

function renderOverview() {
  const s = state.stats || {};
  const h = state.health || {};
  const b = h.backends || {};
  const confirmed = state.slicks.filter((f) => f.properties.is_oil).length;
  const corroborated = state.slicks.filter((f) => f.properties.corroborated).length;

  $("panel-title").textContent = "Overview";
  $("panel-sub").innerHTML =
    `${s.scenes_analysed || 0} scene(s) analysed &middot; ${state.incidents.length} documented spills on the map`;

  const reasons = s.rejection_reasons || {};
  const maxReason = Math.max(1, ...Object.values(reasons));

  $("panel-body").innerHTML = `
    <div class="stat-grid">
      <div class="stat accent">
        <div class="stat-value">${confirmed}</div>
        <div class="stat-label">Slicks confirmed</div>
      </div>
      <div class="stat">
        <div class="stat-value">${s.lookalikes_rejected ?? 0}</div>
        <div class="stat-label">Look-alikes rejected</div>
      </div>
      <div class="stat good">
        <div class="stat-value">${(state.incidents.length).toLocaleString()}</div>
        <div class="stat-label">Documented spills</div>
      </div>
      <div class="stat warn">
        <div class="stat-value">${Math.round((s.abstention_rate ?? 0) * 100)}%</div>
        <div class="stat-label">Abstained</div>
      </div>
    </div>

    ${Object.keys(reasons).length ? `
    <div class="section">
      <h3>Why look-alikes were rejected</h3>
      ${Object.entries(reasons).map(([reason, n]) => `
        <div class="reason-row">
          <div>
            ${esc(reason)}
            <div class="reason-bar" style="width:${(n / maxReason) * 100}%"></div>
          </div>
          <span class="reason-count">${n}</span>
        </div>`).join("")}
    </div>` : ""}

    <div class="section">
      <h3>System state</h3>
      <dl class="kv">
        <dt>Segmentation</dt><dd class="mono" title="${esc(b.segmentation || "")}">${
          esc((b.segmentation || "--").split(" (")[0])}</dd>
        <dt>Drift model</dt><dd class="mono">${esc(b.drift || "--")}</dd>
        <dt>Wind source</dt><dd class="mono">${esc(fieldSummary("wind_sources", b.wind))}</dd>
        <dt>Currents</dt><dd class="mono">${esc(fieldSummary("currents_sources", b.currents))}</dd>
        <dt>GPU</dt><dd class="mono">${b.cuda ? "available" : "CPU only"}</dd>
        <dt>Registry</dt><dd>${(s.documented_incidents ?? 0).toLocaleString()} incidents</dd>
        <dt>Corroborated</dt><dd>${s.corroborated_by_registry ?? 0} detections</dd>
      </dl>
    </div>

    ${(s.abstention_rate ?? 0) > 0.9 ? `
    <div class="notice caution" style="margin-top:0;margin-bottom:14px">
      <b>Almost everything abstains right now.</b> These scenes have no AIS
      coverage loaded, so there is nobody to rank against the drift origin.
      That is a missing input, not an uncertain model &mdash; run
      <code>scripts/fetch_ais.py</code> to supply it.
    </div>` : ""}

    <div class="notice caution">
      <b>Honest limits.</b> Imagery arrives 3&ndash;24 h after acquisition and free
      AIS lags about 72 h &mdash; this is near-real-time, not live. Oil is only
      reliably visible on radar between roughly 2&ndash;3 and 7&ndash;12 m/s of wind.
      Revisit is 6&ndash;12 days, so a spill can appear and disperse between passes.
      SAR cannot measure oil thickness, volume or type.
    </div>
  `;
}

function renderActive() {
  const active = activeSlicks();
  const window_h = fmt(state.slicksMeta.active_window_hours || 72, 0);
  $("panel-title").textContent = "Active detections";
  $("panel-sub").innerHTML = active.length
    ? `${active.length} slick(s) detected within the last ${window_h} hours`
    : "No slicks in currently-fresh imagery";

  if (!active.length) {
    const past = pastSlicks().length;
    $("panel-body").innerHTML = `
      <div class="notice caution" style="margin-top:0">
        <b>Nothing active right now.</b> A detection counts as active only while
        a present position can honestly be forecast &mdash; about ${window_h} hours
        after acquisition. Beyond that the oil has dispersed, stranded or
        weathered, so it is filed as a past incident rather than left on the map
        implying it is still out there.
      </div>
      ${unattributedNotice()}
      <div class="empty">
        ${past} past detection(s) available under <b>Past incidents</b>.<br><br>
        For active detections, fetch recent imagery:
        <code>python scripts/fetch_sentinel.py --config configs/fetch_elsa3.yaml</code>
      </div>`;
    return;
  }

  $("panel-body").innerHTML =
    `<div class="section"><h3>Detected in fresh imagery</h3>
      ${active.map(slickCard).join("")}</div>
     ${unattributedNotice()}
     <div class="notice caution">${esc(state.slicksMeta.disclaimer || "")}</div>`;
  wireCards();
}

function unattributedNotice() {
  const n = unattributedSlicks().length;
  if (!n) return "";
  return `<div class="notice caution">
    <b>${n} further detection(s) have insufficient data to attribute.</b>
    The physics stage confirmed them as oil, but no AIS track could be ranked
    against the drift origin &mdash; so no vessel is suggested at all. They are
    listed separately under <b>Insufficient data</b> rather than here, because
    a detection with no candidate is not a lead.</div>`;
}

function renderPast() {
  const past = pastSlicks();
  const rejected = state.slicks.filter((f) => !f.properties.is_oil);
  state.pastFilter = state.pastFilter || "detections";

  $("panel-title").textContent = "Past incidents";
  $("panel-sub").innerHTML =
    `${past.length} of our detections &middot; ${state.incidents.length} from public registries`;

  const unattributed = unattributedSlicks();
  const filters = [
    ["detections", "Our detections (" + past.length + ")"],
    ["insufficient", "Insufficient data (" + unattributed.length + ")"],
    ["documented", "Documented (" + state.incidents.length + ")"],
    ["rejected", "Rejected (" + rejected.length + ")"],
  ];

  let body = '<div class="filter-row">' + filters.map(([key, label]) =>
    '<button class="filter-btn ' + (state.pastFilter === key ? "on" : "") +
    '" data-filter="' + key + '">' + label + "</button>").join("") + "</div>";

  if (state.pastFilter === "detections") {
    body += past.length
      ? '<div class="section">' + past.map(slickCard).join("") + "</div>"
      : '<div class="empty">No past detections yet.</div>';
    body += `<div class="notice caution">
      These are slicks OUR pipeline found in archived imagery. They are
      historical &mdash; that oil is long gone. Shown so every detection can be
      inspected and audited.</div>`;
  } else if (state.pastFilter === "insufficient") {
    body += unattributed.length
      ? '<div class="section">' + unattributed.map(slickCard).join("") + "</div>"
      : '<div class="empty">Every current detection has a ranked candidate.</div>';
    body += `<div class="notice caution">
      <b>Confirmed as oil, but unattributable.</b> Each of these passed the
      physics stage &mdash; wind, damping, shape and texture. What is missing is
      AIS: with no vessel track near the drift-estimated origin there is nobody
      to rank, so the system declines to name one. Absence of a ship is not
      evidence against oil &mdash; a wreck, a natural seep, or a vessel running
      dark all look exactly like this.</div>`;
  } else if (state.pastFilter === "documented") {
    const sorted = [...state.incidents].sort((a, b) => {
      const va = a.properties.volume_m3 || 0, vb = b.properties.volume_m3 || 0;
      if (vb !== va) return vb - va;
      return String(b.properties.occurred_at || "").localeCompare(
        String(a.properties.occurred_at || ""));
    });
    body += `<div class="notice confirmed" style="margin-top:0">
      <b>Confirmed events, not detections.</b> Every entry is a real spill
      recorded by a public registry, independent of our model.</div>` +
      '<div class="section">' + sorted.slice(0, 60).map(incidentCard).join("") + "</div>";
  } else {
    body += rejected.length
      ? '<div class="section">' + rejected.map(slickCard).join("") + "</div>"
      : '<div class="empty">Nothing was rejected.</div>';
    body += `<div class="notice caution">
      Dark patches the physics stage ruled out, kept with their reasons so every
      rejection can be checked.</div>`;
  }

  $("panel-body").innerHTML = body;
  $("panel-body").querySelectorAll("[data-filter]").forEach((el) =>
    el.addEventListener("click", () => {
      state.pastFilter = el.dataset.filter;
      renderPast();
    }));
  wireCards();
  $("panel-body").querySelectorAll("[data-ilat]").forEach((el) =>
    el.addEventListener("click", () =>
      focusPoint(Number(el.dataset.ilat), Number(el.dataset.ilon), 8)));
}

function wireCards() {
  $("panel-body").querySelectorAll("[data-cid]").forEach((el) =>
    el.addEventListener("click", () => {
      if (el.dataset.oil === "1") openDetail(el.dataset.cid);
      else focusPoint(Number(el.dataset.lat), Number(el.dataset.lon));
    }));
}

function slickCard(f) {
  const p = f.properties;
  const [lat, lon] = centroidOf(f);
  const isPast = p.activity === "historical";
  const tier = p.confidence_tier;
  const pill = !p.is_oil
    ? '<span class="pill rejected">Rejected</span>'
    : tier === "confirmed"
      ? '<span class="pill confirmed">Confirmed oil</span>'
      : tier === "probable"
        ? '<span class="pill oil">Probable oil</span>'
        : '<span class="pill abstain">Possible</span>';
  const top = p.top_candidate;

  return '<div class="slick-card ' + (p.is_oil ? "" : "rejected") + " " +
    (isPast && p.is_oil ? "past" : "") + '"' +
    ' data-cid="' + esc(p.candidate_id) + '" data-oil="' + (p.is_oil ? 1 : 0) + '"' +
    ' data-lat="' + lat + '" data-lon="' + lon + '">' +
    '<div class="card-top">' +
      '<span class="card-id">' + esc(p.candidate_id) + "</span>" +
      '<span style="display:flex;gap:5px">' +
        (isPast && p.is_oil ? '<span class="pill past">past</span>' : "") + pill +
      "</span>" +
    "</div>" +
    '<div class="card-metrics">' +
      "<span><b>" + fmt(p.area_km2) + "</b> km&sup2;</span>" +
      "<span>P(oil) <b>" + fmt(p.p_oil) + "</b></span>" +
      "<span>wind <b>" + fmt((p.wind || {}).speed_ms, 1) + "</b> m/s</span>" +
      (p.age_days != null ? "<span>" + fmt(p.age_days, 0) + " d ago</span>" : "") +
    "</div>" +
    (top ? '<div class="card-suspect">Top candidate: <b>' +
      esc(top.name || top.mmsi) + "</b> &middot; " + fmt(top.score) +
      (top.went_dark ? '<span class="pill dark">went dark</span>' : "") + "</div>" : "") +
    (!p.is_oil && p.rejected_reason
      ? '<div class="card-reason">' + esc(p.rejected_reason) + "</div>" : "") +
    "</div>";
}

function incidentCard(f) {
  const p = f.properties;
  const [lon, lat] = f.geometry.coordinates;
  return `<div class="slick-card" data-ilat="${lat}" data-ilon="${lon}">
    <div class="card-top">
      <span style="font-weight:610;font-size:12.5px">${esc(p.name).slice(0, 46)}</span>
      ${p.persistent ? '<span class="pill dark">persistent</span>'
        : p.natural_seep ? '<span class="pill rejected">natural seep</span>'
        : '<span class="pill confirmed">documented</span>'}
    </div>
    <div class="card-metrics">
      <span>${esc(p.location || "").slice(0, 40) || "&mdash;"}</span>
    </div>
    <div class="card-metrics">
      <span>${p.persistent ? "ongoing" : fmtDate(p.occurred_at)}</span>
      ${p.commodity ? `<span>${esc(p.commodity).slice(0, 26)}</span>` : ""}
      ${p.volume_m3 ? `<span><b>${Math.round(p.volume_m3).toLocaleString()}</b> m&sup3;</span>` : ""}
    </div>
  </div>`;
}

function focusPoint(lat, lon, zoom = 9) {
  state.map.flyTo([lat, lon], zoom, { duration: 0.7 });
}

/* ------------------------------------------------------------- detail --- */

async function openDetail(candidateId) {
  $("panel-body").innerHTML = '<div class="spinner">Backtracking drift</div>';
  let detail, trace;
  try {
    detail = await fetch(apiUrl(`/api/slicks/${candidateId}`)).then((r) => r.json());
    const [traceRes, tlRes] = await Promise.all([
      fetch(apiUrl(`/api/slicks/${candidateId}/backtrace`)),
      fetch(apiUrl(`/api/slicks/${candidateId}/timeline`)),
    ]);
    trace = traceRes.ok ? await traceRes.json() : null;
    state.timeline = tlRes.ok ? await tlRes.json() : null;
  } catch (err) {
    $("panel-body").innerHTML =
      `<div class="empty">Could not load ${esc(candidateId)}<br><br>${esc(err)}</div>`;
    return;
  }
  state.current = { detail, trace };
  state.fromDetail = true;
  renderDetailPanel(detail, trace);
  drawDetailMap(detail, trace);
}

function renderDetailPanel(d, trace) {
  const wind = d.wind || {};
  const origin = d.origin;
  const corr = (d.evidence || {}).corroboration;
  const conf = (d.evidence || {}).confidence;

  $("panel-title").textContent = d.candidate_id;
  $("panel-sub").textContent = d.abstained
    ? "Insufficient evidence to rank a source"
    : `${d.vessels.length} candidate vessel(s), ranked by correlation`;

  // The synthetic scene exists only to demonstrate vessel ranking, which no
  // real scene here can show: the Indian Ocean passes have no AIS coverage,
  // MC-20 is correctly routed to infrastructure, and the Galveston pass had no
  // slick. It must announce itself before any of its numbers are read.
  const synthNotice = d.synthetic ? `
      <div class="notice synthetic" style="margin-top:0">
        <b>SYNTHETIC DEMONSTRATION &mdash; not a real detection.</b>
        The imagery, the slick and the AIS tracks below are all fabricated, and
        exist to show how vessel ranking works. Every other scene on this map is
        real Sentinel-1 data with real ERA5 wind and real CMEMS currents. Never
        quote a number from this scene.
      </div>` : "";

  $("panel-body").innerHTML = `
    <button class="back-link" id="back-btn">&larr; Back</button>
    ${synthNotice}

    ${corr && corr.confirmed ? `
      <div class="notice confirmed">
        <b>Corroborated by an independent registry.</b>
        ${esc((corr.matches[0] || {}).reason || "")}
      </div>` : ""}

    ${d.abstained ? `
      <div class="section"><h3>Abstention</h3>
        <div class="card-reason" style="border-left-color:var(--abstain)">
          ${esc(d.abstain_reason)}</div></div>` : ""}

    ${timelineHTML(state.timeline)}

    ${conf ? `
    <div class="section">
      <h3>Is this actually oil?</h3>
      <div class="notice ${conf.tier === "confirmed" ? "confirmed" : conf.tier === "probable" ? "caution" : "synthetic"}"
           style="margin-top:0">
        <b>${esc(conf.tier.toUpperCase())}</b> &middot; score ${fmt(conf.score)}<br>
        ${esc(conf.meaning)}
      </div>
      <ul style="margin:9px 0 0;padding-left:17px;font-size:10.5px;line-height:1.65;color:var(--muted)">
        ${(conf.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("")}
      </ul>
    </div>` : ""}

    <div class="section">
      <h3>Physics check</h3>
      <dl class="kv">
        <dt>P(oil)</dt><dd>${fmt(d.slick.p_oil)}</dd>
        <dt>Area</dt><dd>${fmt(d.slick.area_km2)} km&sup2;</dd>
        <dt>Morphology</dt><dd>${esc(d.slick.morphology)}</dd>
        <dt>Wind speed</dt><dd>${fmt(wind.speed_ms, 1)} m/s</dd>
        <dt>Wind window</dt><dd>${fmt(wind.window_score)}</dd>
        <dt>Wind source</dt><dd class="mono">${esc(wind.source)}</dd>
      </dl>
      ${contributionsHTML((d.evidence || {}).physics_contributions)}
    </div>

    ${origin ? `
    <div class="section">
      <h3>Estimated origin &mdash; backward drift</h3>
      <dl class="kv">
        <dt>Position</dt><dd class="mono">${fmt(origin.lat, 4)}&deg;N ${fmt(origin.lon, 4)}&deg;E</dd>
        <dt>Released at</dt><dd>${fmtTime(origin.estimated_at)}</dd>
        <dt>Uncertainty</dt><dd>&plusmn;${fmt(origin.uncertainty_km, 1)} km</dd>
        <dt>Backtracked</dt><dd>${fmt(origin.backtrack_hours, 1)} h</dd>
        <dt>Particles</dt><dd>${origin.n_particles}</dd>
        <dt>Method</dt><dd class="mono">${esc(origin.method)}</dd>
      </dl>
      ${trace ? playbarHTML(trace) : ""}
      ${!origin.reliable ? `<div class="card-reason" style="border-left-color:var(--abstain);margin-top:9px">
        Beyond ~24 h of backtracking the origin is a wide blur, not a point.</div>` : ""}
    </div>` : ""}

    <div class="section">
      <h3>Candidate vessels &mdash; ranked, not accused</h3>
      ${d.vessels.length
        ? d.vessels.map(vesselHTML).join("")
        : '<div class="empty">No vessel tracks near the estimated origin.</div>'}
    </div>

    <div class="notice caution">${esc(d.disclaimer)}</div>`;

  $("back-btn").addEventListener("click", () => {
    drawSlickLayers();
    drawIncidentLayer();
    renderTab("slicks");
  });
  $("panel-body").querySelectorAll("[data-mmsi]").forEach((el) =>
    el.addEventListener("click", () => focusVessel(el.dataset.mmsi)));
  if (trace) wirePlaybar(trace);
  wireTimeline();
}

function timelineHTML(tl) {
  /* Three positions of the same slick: where it began, what the satellite
     recorded, and where it probably is now. The middle one is the only
     observation; the outer two are model estimates and say so. */
  if (!tl || !tl.states || !tl.states.length) return "";
  const byLabel = Object.fromEntries(tl.states.map((s) => [s.label, s]));
  const nowState = byLabel.now || byLabel.historical;
  const steps = [
    { key: "origin", name: "Origin", glyph: "◀", state: byLabel.origin },
    { key: "observed", name: "Observed", glyph: "●", state: byLabel.observed },
    {
      key: nowState && nowState.label === "historical" ? "historical" : "now",
      name: nowState && nowState.label === "historical" ? "Dispersed" : "Now",
      glyph: "▶", state: nowState,
    },
  ];

  return '<div class="timeline"><div class="timeline-track">' +
    steps.map((s, i) =>
      '<button class="tl-step ' + (s.state ? "" : "tl-disabled") +
      (i === 1 ? " active" : "") + '" data-label="' + s.key + '"' +
      (s.state ? "" : " disabled") + ">" +
        '<div class="tl-dot">' + s.glyph + "</div>" +
        '<div class="tl-name">' + s.name + "</div>" +
        '<div class="tl-time">' + (s.state ? fmtDate(s.state.at) : "&mdash;") + "</div>" +
      "</button>").join("") +
    "</div>" +
    '<div class="tl-detail" id="tl-detail">' + stepDetail(byLabel.observed) + "</div>" +
    "</div>";
}

function stepDetail(st) {
  if (!st) return "No estimate available for this point in time.";
  const unc = st.uncertainty_km > 0
    ? ` &plusmn;${fmt(st.uncertainty_km, 1)} km` : " (observed directly)";
  return "<b>" + fmt(st.lat, 4) + "&deg;N " + fmt(st.lon, 4) + "&deg;E</b>" + unc +
    "<br>" + fmtTime(st.at) + "<br>" + esc(st.description);
}

function wireTimeline() {
  const tl = state.timeline;
  if (!tl) return;
  const byLabel = Object.fromEntries(tl.states.map((s) => [s.label, s]));
  document.querySelectorAll(".tl-step").forEach((btn) => {
    btn.addEventListener("click", () => {
      const st = byLabel[btn.dataset.label];
      if (!st) return;
      document.querySelectorAll(".tl-step").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const detail = $("tl-detail");
      if (detail) detail.innerHTML = stepDetail(st);
      showTimelineState(st, tl);
    });
  });
}

function showTimelineState(st, tl) {
  /* Move the map to the chosen moment and keep the vessel routes visible,
     so "where did it come from" and "who was there" stay on screen together. */
  state.map.flyTo([st.lat, st.lon], Math.max(state.map.getZoom(), 7), { duration: 0.6 });
  if (state.tlMarker) state.layers.detail.removeLayer(state.tlMarker);
  if (state.tlCircle) state.layers.detail.removeLayer(state.tlCircle);

  const colour = st.label === "origin" ? C.origin
    : st.label === "observed" ? C.oil : C.vessel;

  if (st.uncertainty_km > 0) {
    state.tlCircle = L.circle([st.lat, st.lon], {
      radius: st.uncertainty_km * 1000, color: colour, weight: 1,
      opacity: 0.55, fillColor: colour, fillOpacity: 0.09,
    }).addTo(state.layers.detail);
  }
  state.tlMarker = L.circleMarker([st.lat, st.lon], {
    radius: 9, color: "#fff", fillColor: colour, fillOpacity: 0.95, weight: 2,
  }).addTo(state.layers.detail)
    .bindTooltip("<b>" + esc(st.label) + "</b><br>" + fmtTime(st.at), { direction: "top" })
    .openTooltip();
}

function contributionsHTML(contrib) {
  if (!contrib || !Object.keys(contrib).length) return "";
  const rows = Object.entries(contrib)
    .filter(([k]) => k !== "_bias")
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  if (!rows.length) return "";
  const max = Math.max(...rows.map(([, v]) => Math.abs(v)), 0.001);
  return `<div style="margin-top:12px">
    <div style="font-size:9px;letter-spacing:.11em;text-transform:uppercase;color:var(--dim);font-weight:650;margin-bottom:7px">
      Evidence weights (log-odds)</div>
    <div class="contrib">${rows.map(([k, v]) => `
      <div class="contrib-row">
        <span style="color:var(--muted)">${esc(k.replace(/_/g, " "))}</span>
        <div class="contrib-bar"><div class="contrib-axis"></div>
          <div class="contrib-fill ${v >= 0 ? "pos" : "neg"}" style="width:${(Math.abs(v) / max) * 50}%"></div>
        </div>
        <span class="bar-val">${v >= 0 ? "+" : ""}${fmt(v)}</span>
      </div>`).join("")}</div></div>`;
}

function vesselHTML(v) {
  return `<div class="vessel rank-${v.rank}" data-mmsi="${esc(v.mmsi)}">
    <div class="vessel-head">
      <div style="min-width:0">
        <div class="vessel-name">#${v.rank} ${esc(v.name || "Unknown vessel")}
          ${v.went_dark ? '<span class="pill dark">went dark</span>' : ""}</div>
        <div class="vessel-mmsi">MMSI ${esc(v.mmsi)}${v.vessel_type ? " &middot; " + esc(v.vessel_type) : ""}${v.flag ? " &middot; " + esc(v.flag) : ""}</div>
      </div>
      <div class="vessel-score">${fmt(v.score)}</div>
    </div>
    <div class="bars">
      ${barHTML("parity", v.parity)}
      ${barHTML("proximity", v.proximity)}
      ${barHTML("temporality", v.temporality)}
    </div>
    ${voyageHTML(v.voyage)}
    <div class="evidence-text">${esc(v.evidence)}</div>
  </div>`;
}

function barHTML(label, value) {
  return `<div class="bar-row">
    <span class="bar-label">${label}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${(value || 0) * 100}%"></div></div>
    <span class="bar-val">${fmt(value)}</span>
  </div>`;
}

function voyageHTML(vy) {
  if (!vy) return "";
  const pa = vy.projected_arrival;
  return `<div class="voyage">
    <div class="voyage-leg">
      <span class="pin start">A</span>
      <div><div class="voyage-place">${esc(vy.from.nearest_port || "open sea")}</div>
        <div class="voyage-time">${fmtTime(vy.from.at)}</div></div>
    </div>
    <div class="voyage-line">
      ${fmt(vy.distance_km, 0)} km &middot; ${fmt(vy.duration_hours, 1)} h &middot; ${fmt(vy.mean_speed_knots, 1)} kn
    </div>
    <div class="voyage-leg">
      <span class="pin end">B</span>
      <div><div class="voyage-place">${esc(vy.to.nearest_port || "open sea")}</div>
        <div class="voyage-time">${fmtTime(vy.to.at)}</div></div>
    </div>
    ${vy.declared_destination
      ? `<div class="voyage-meta">Declared destination (AIS): <b>${esc(vy.declared_destination)}</b></div>` : ""}
    ${pa && pa.port
      ? `<div class="voyage-meta">Course points toward <b>${esc(pa.port)}</b>, ~${fmt(pa.hours, 1)} h on &mdash; projection, not a declared route</div>` : ""}
    <div class="voyage-note">${esc(vy.coverage_note)}</div>
  </div>`;
}

/* -------------------------------------------------------- detail map --- */

function drawDetailMap(d, trace) {
  clearDetail();
  state.map.removeLayer(state.layers.slicks);
  state.map.removeLayer(state.layers.rejected);
  const bounds = [];

  function addEndpoint(point, kind, color, primary, html) {
    L.marker([point.lat, point.lon], {
      icon: L.divIcon({
        className: "",
        html: `<div class="route-pin ${kind}" style="--pin:${color}">${kind === "start" ? "A" : "B"}</div>`,
        iconSize: [18, 18], iconAnchor: [9, 9],
      }),
      opacity: primary ? 1 : 0.75,
    }).addTo(state.layers.detail).bindTooltip(html, { direction: "top" });
    bounds.push(L.latLngBounds([[point.lat, point.lon]]));
  }

  if (d.slick.polygon && d.slick.polygon.length >= 4) {
    const poly = L.polygon(d.slick.polygon.map(([lon, lat]) => [lat, lon]), {
      color: C.oil, weight: 2, fillColor: C.oil, fillOpacity: 0.32,
    }).addTo(state.layers.detail);
    poly.bindTooltip("Observed slick", { direction: "top" });
    bounds.push(poly.getBounds());
  }

  if (d.origin) {
    const track = (d.origin.track || []).map((p) => [p.lat, p.lon]);
    if (track.length > 1) {
      L.polyline(track, { color: C.origin, weight: 2.5, opacity: 0.85, dashArray: "6,5" })
        .addTo(state.layers.detail)
        .bindTooltip("Backward drift path", { sticky: true });
    }
    L.circle([d.origin.lat, d.origin.lon], {
      radius: d.origin.uncertainty_km * 1000,
      color: C.origin, weight: 1, opacity: 0.5,
      fillColor: C.origin, fillOpacity: 0.08,
    }).addTo(state.layers.detail)
      .bindTooltip(`Origin uncertainty ±${fmt(d.origin.uncertainty_km, 1)} km`, { sticky: true });

    L.circleMarker([d.origin.lat, d.origin.lon], {
      radius: 8, color: C.origin, fillColor: C.origin, fillOpacity: 0.95, weight: 2,
    }).addTo(state.layers.detail)
      .bindTooltip(
        `<b>Estimated origin</b><br>${fmtTime(d.origin.estimated_at)}<br>±${fmt(d.origin.uncertainty_km, 1)} km`,
        { direction: "top" }
      ).openTooltip();
    bounds.push(L.latLngBounds([[d.origin.lat, d.origin.lon]]));
  }

  d.vessels.forEach((v) => {
    if (!v.track || v.track.length < 2) return;
    const color = v.went_dark ? C.dark : C.vessel;
    const primary = v.rank === 1;
    const segments = splitOnGaps(v.track);

    segments.forEach((seg) => {
      const line = L.polyline(seg.map((p) => [p.lat, p.lon]), {
        color, weight: primary ? 3.5 : 2, opacity: primary ? 0.95 : 0.55,
      }).addTo(state.layers.detail);
      line.bindTooltip(vesselTooltip(v), { sticky: true });
      line._mmsi = v.mmsi;
      bounds.push(line.getBounds());
    });

    // Draw the AIS silence as an explicit gap, never a smooth line.
    for (let i = 0; i < segments.length - 1; i++) {
      const a = segments[i][segments[i].length - 1];
      const b = segments[i + 1][0];
      L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {
        color, weight: 2, opacity: 0.9, dashArray: "3,7",
      }).addTo(state.layers.detail)
        .bindTooltip("AIS silent across this leg", { sticky: true });
    }

    const vy = v.voyage;
    const first = v.track[0];
    const last = v.track[v.track.length - 1];
    addEndpoint(first, "start", color, primary,
      `<b>${esc(v.name || v.mmsi)}</b><br>Track begins${vy && vy.from.nearest_port ? " near " + esc(vy.from.nearest_port) : ""}<br>${fmtTime(first.at)}`);
    addEndpoint(last, "end", color, primary,
      `<b>${esc(v.name || v.mmsi)}</b><br>Track ends${vy && vy.to.nearest_port ? " near " + esc(vy.to.nearest_port) : ""}<br>${fmtTime(last.at)}` +
      (vy && vy.declared_destination ? `<br>Declared: ${esc(vy.declared_destination)}` : ""));

    if (primary && vy && vy.projected_arrival && vy.projected_arrival.lon != null) {
      const pa = vy.projected_arrival;
      L.polyline([[last.lat, last.lon], [pa.lat, pa.lon]], {
        color, weight: 1.5, opacity: 0.45, dashArray: "2,8",
      }).addTo(state.layers.detail)
        .bindTooltip(
          `Projected onward course${pa.port ? " toward " + esc(pa.port) : ""}` +
          `${pa.hours ? " (~" + fmt(pa.hours, 1) + " h)" : ""}<br><i>${esc(pa.basis)}</i>`,
          { sticky: true });
    }
  });

  if (bounds.length) {
    state.map.fitBounds(bounds.reduce((a, b) => a.extend(b)), { padding: [70, 70] });
  }
  if (trace) setupAnimation(trace);
}

function vesselTooltip(v) {
  const vy = v.voyage;
  let html = `<b>#${v.rank} ${esc(v.name || v.mmsi)}</b><br>score ${fmt(v.score)}`;
  if (vy) {
    html += `<br>${esc(vy.from.nearest_port || "open sea")} &rarr; ${esc(vy.to.nearest_port || "open sea")}`;
    html += `<br>${fmt(vy.distance_km, 0)} km over ${fmt(vy.duration_hours, 1)} h at ${fmt(vy.mean_speed_knots, 1)} kn`;
  }
  html += `<br>closest ${fmt(v.closest_approach_km, 1)} km at ${fmtTime(v.closest_approach_at)}`;
  if (v.went_dark) html += "<br><b>AIS gap at the origin</b>";
  return html;
}

function splitOnGaps(track, gapMinutes = 45) {
  /* Break a track at AIS silences so a gap is drawn as a gap, not a straight
     line implying the vessel was tracked the whole way. */
  const segments = [];
  let current = [track[0]];
  for (let i = 1; i < track.length; i++) {
    const dt = (new Date(track[i].at) - new Date(track[i - 1].at)) / 60000;
    if (dt > gapMinutes) { segments.push(current); current = []; }
    current.push(track[i]);
  }
  if (current.length) segments.push(current);
  return segments.filter((s) => s.length > 1);
}

function focusVessel(mmsi) {
  document.querySelectorAll(".vessel").forEach((el) =>
    el.classList.toggle("active", el.dataset.mmsi === mmsi));
  state.layers.detail.eachLayer((layer) => {
    if (layer._mmsi === mmsi && layer.getBounds) {
      state.map.fitBounds(layer.getBounds(), { padding: [80, 80] });
      layer.setStyle({ weight: 5, opacity: 1 });
    } else if (layer._mmsi) {
      layer.setStyle({ weight: 2, opacity: 0.35 });
    }
  });
}

/* ---------------------------------------------------------- animation --- */

function playbarHTML(trace) {
  return `<div class="playbar">
    <button class="playbtn" id="play-btn" title="Play the backward drift">&#9654;</button>
    <input type="range" class="scrub" id="scrub" min="0" max="${Math.max(trace.n_frames - 1, 0)}" value="0" />
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
    radius: 7, color: "#fff", fillColor: C.oil, fillOpacity: 0.95, weight: 2,
  }).addTo(state.layers.detail);
  state.anim.trail = L.polyline([[first.lat, first.lon]], {
    color: C.oil, weight: 3, opacity: 0.75,
  }).addTo(state.layers.detail);
  showFrame(0);
}

function wirePlaybar() {
  const btn = $("play-btn"), scrub = $("scrub");
  if (!btn || !scrub) return;
  btn.addEventListener("click", () => {
    if (state.anim.timer) { stopAnimation(); btn.innerHTML = "&#9654;"; }
    else { startAnimation(); btn.innerHTML = "&#10074;&#10074;"; }
  });
  scrub.addEventListener("input", (e) => {
    stopAnimation(); btn.innerHTML = "&#9654;";
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
  if (state.anim.trail)
    state.anim.trail.setLatLngs(frames.slice(0, idx + 1).map((p) => [p.lat, p.lon]));
  const label = $("frame-time");
  if (label) {
    label.innerHTML = `${fmtTime(f.at)}<br>${f.hours_before > 0 ? "&minus;" + fmt(f.hours_before, 1) + " h" : "observed"}`;
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
  if (state.anim.timer) { clearInterval(state.anim.timer); state.anim.timer = null; }
}

/* --------------------------------------------------------------- init --- */

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initResponsivePanel();
  $("tabs").addEventListener("click", (e) => {
    const tab = e.target.closest(".tab");
    if (!tab) return;
    // Leaving a detail view must restore whatever layers it hid.
    if (!state.map.hasLayer(state.layers.slicks) && $("layer-slicks").checked) {
      state.layers.slicks.addTo(state.map);
    }
    if (!state.map.hasLayer(state.layers.rejected) && $("layer-rejected").checked) {
      state.layers.rejected.addTo(state.map);
    }
    renderTab(tab.dataset.tab);
  });
  boot();
});

/* ----------------------------------------------------------- responsive --- */

function initResponsivePanel() {
  /* On a stacked layout the panel competes with the map for a small screen.
     Dragging its header resizes it, and a tap toggles collapsed/expanded, so
     the map can be given the whole viewport when you are looking at it. */
  const panel = $("panel");
  const head = document.querySelector(".panel-head");
  if (!panel || !head) return;

  const stacked = () => window.matchMedia("(max-width: 1180px)").matches;
  let startY = 0;
  let startHeight = 0;
  let dragging = false;
  let moved = false;

  const onDown = (e) => {
    if (!stacked()) return;
    // Let the tabs work; only the bare header area is a drag handle.
    if (e.target.closest(".tab") || e.target.closest("button")) return;
    dragging = true;
    moved = false;
    startY = (e.touches ? e.touches[0].clientY : e.clientY);
    startHeight = panel.getBoundingClientRect().height;
    panel.style.transition = "none";
  };

  const onMove = (e) => {
    if (!dragging) return;
    const y = (e.touches ? e.touches[0].clientY : e.clientY);
    const delta = startY - y;
    if (Math.abs(delta) > 4) moved = true;
    const height = Math.min(
      window.innerHeight * 0.88,
      Math.max(92, startHeight + delta)
    );
    panel.style.maxHeight = `${height}px`;
    if (e.cancelable) e.preventDefault();
  };

  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    panel.style.transition = "";
    if (!moved) {
      // A tap, not a drag: toggle between peek and full.
      panel.classList.toggle("collapsed");
      panel.style.maxHeight = "";
    }
    state.map.invalidateSize();
  };

  head.addEventListener("mousedown", onDown);
  head.addEventListener("touchstart", onDown, { passive: true });
  window.addEventListener("mousemove", onMove);
  window.addEventListener("touchmove", onMove, { passive: false });
  window.addEventListener("mouseup", onUp);
  window.addEventListener("touchend", onUp);

  // Leaflet needs telling when its container changes size, or the tiles tear.
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      panel.style.maxHeight = "";
      state.map.invalidateSize();
      applyZoomStyling();
    }, 140);
  });
  window.addEventListener("orientationchange", () =>
    setTimeout(() => state.map.invalidateSize(), 260));
}
