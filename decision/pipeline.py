"""The full pipeline, stage by stage.

    ingest -> stage_a -> stage_b -> lookalike -> morphology
           -> drift -> AIS -> scoring -> decision

Every stage records its own elapsed time so the latency breakdown in the
evaluation report comes from real measurements rather than guesswork.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from core.config import Config, resolve_path
from core.contracts import (
    Attribution,
    Scene,
    SlickCandidate,
    WindContext,
)
from decision.rank import DecisionConfig, decide
from detect.lookalike import LookalikeModel
from detect.morphology import classify_morphology
from detect.polygonize import RegionFeatures, extract_regions, polygon_wkt

log = logging.getLogger(__name__)


@dataclass
class SceneAnalysis:
    """Everything produced for one scene."""

    scene: Scene
    candidates: list[SlickCandidate]
    attributions: list[Attribution]
    rejected: list[SlickCandidate]
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def confirmed(self) -> list[SlickCandidate]:
        return [c for c in self.candidates if not c.is_rejected]


class Stopwatch:
    """Per-stage timings, so the latency table is measured not estimated."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._start: float | None = None
        self._label: str | None = None

    def __call__(self, label: str) -> Stopwatch:
        self._label = label
        return self

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._label and self._start is not None:
            self.timings[self._label] = round(time.perf_counter() - self._start, 3)
        return False


def region_to_candidate(
    region: RegionFeatures, scene: Scene, wind: WindContext, verdict, morph
) -> SlickCandidate:
    """Assemble the frozen contract object from the measured region."""
    return SlickCandidate(
        candidate_id=f"{scene.scene_id}-{region.label:03d}",
        scene_id=scene.scene_id,
        polygon_wkt=polygon_wkt(region.polygon_lonlat),
        area_km2=round(region.area_km2, 4),
        elongation=round(region.elongation, 3),
        compactness=round(region.compactness, 4),
        damping_ratio=round(region.damping_ratio, 5) if region.damping_ratio == region.damping_ratio else float("nan"),
        wind=wind,
        p_oil=verdict.p_oil,
        rejected_reason=verdict.rejected_reason,
        morphology=morph.morphology,
        centroid=region.centroid_lonlat,
        texture_homogeneity=round(region.texture_homogeneity, 4),
        texture_contrast=round(region.texture_contrast, 4),
        texture_variance=round(region.texture_variance, 4),
        vh_vv_ratio=round(region.vh_vv_ratio, 5) if region.vh_vv_ratio is not None else None,
        feature_contributions={k: round(v, 4) for k, v in verdict.contributions.items()},
    )


def analyse_scene(
    scene: Scene,
    config: Config,
    wind_lookup=None,
    ais_tracks=None,
) -> SceneAnalysis:
    """Run every stage for one scene.

    wind_lookup(lon, lat, when) -> WindContext. Required: CLAUDE.md rule 3
    says a candidate without wind context is not a candidate, so there is no
    default that quietly invents one.
    """
    from ingest.pipeline import ingest_scene
    from detect.stage_a import screen_tiles
    from detect.stage_b import run_stage_b

    sw = Stopwatch()
    warnings: list[str] = []

    with sw("ingest"):
        ingest = ingest_scene(scene, config)

    with sw("stage_a"):
        keep = None
        if not config.use_stub("stage_a"):
            screen = screen_tiles(
                ingest.grid,
                threshold=float(config.get("detect.stage_a_threshold", 0.15)),
            )
            keep = screen.keep_indices
            if not keep:
                log.info("Stage A found nothing worth segmenting in %s", scene.scene_id)

    with sw("stage_b"):
        seg = run_stage_b(ingest, config, keep_indices=keep)

    with sw("polygonize"):
        regions = extract_regions(
            probability=seg.oil_probability,
            sigma0_db=ingest.sigma0_db,
            light_db=ingest.light_db,
            land_mask=ingest.land_mask,
            bbox=ingest.bbox,
            resolution_m=ingest.grid.resolution_m,
            threshold=float(config.get("detect.stage_b_threshold", 0.5)),
            min_area_km2=float(config.get("detect.min_area_km2", 0.05)),
        )

    if wind_lookup is None:
        raise ValueError(
            "analyse_scene requires wind_lookup. A candidate without wind "
            "context is not a candidate (CLAUDE.md rule 3)."
        )

    lookalike = LookalikeModel.load_or_prior(
        resolve_path(config.get("lookalike.model_path", "models/lookalike.json")),
        threshold=float(config.get("lookalike.p_oil_threshold", 0.5)),
    )

    candidates: list[SlickCandidate] = []
    rejected: list[SlickCandidate] = []
    morphologies: dict[str, Any] = {}
    region_by_id: dict[str, RegionFeatures] = {}

    # Scene-level check: internal waves arrive as a train of evenly spaced
    # parallel bands. Judged individually every band passes as a thin streak,
    # so this is the one test that needs the whole scene at once.
    from detect.wavetrain import wave_train_rejections

    wave_rejects = wave_train_rejections(regions)
    if wave_rejects:
        log.info(
            "Internal-wave trains account for %d of %d regions",
            len(wave_rejects), len(regions),
        )

    with sw("lookalike"):
        for region in regions:
            lon, lat = region.centroid_lonlat
            wind = wind_lookup(lon, lat, scene.acquired_at)
            verdict = lookalike.classify(region, wind)
            morph = classify_morphology(region)

            wave_reason = wave_rejects.get(region.label)
            if wave_reason is not None and verdict.rejected_reason is None:
                verdict.rejected_reason = wave_reason
                verdict.gate_hit = "internal_wave_train"
                verdict.p_oil = min(verdict.p_oil, 0.2)

            cand = region_to_candidate(region, scene, wind, verdict, morph)
            morphologies[cand.candidate_id] = morph
            region_by_id[cand.candidate_id] = region
            candidates.append(cand)
            if cand.is_rejected:
                rejected.append(cand)
                log.info("REJECTED %s: %s", cand.candidate_id, cand.rejected_reason)

    confirmed = [c for c in candidates if not c.is_rejected]
    log.info(
        "%s: %d regions -> %d confirmed oil, %d rejected as look-alikes",
        scene.scene_id, len(regions), len(confirmed), len(rejected),
    )

    attributions: list[Attribution] = []
    dcfg = DecisionConfig(
        abstain_margin=float(config.get("decision.abstain_margin", 0.08)),
        min_top_score=float(config.get("decision.min_top_score", 0.35)),
        min_wind_window_score=float(config.get("decision.min_wind_window_score", 0.15)),
    )

    with sw("drift_and_attribute"):
        for cand in candidates:
            morph = morphologies[cand.candidate_id]
            region = region_by_id[cand.candidate_id]
            attributions.append(
                _attribute_one(cand, region, morph, scene, config, dcfg, ais_tracks, warnings)
            )

    return SceneAnalysis(
        scene=scene,
        candidates=candidates,
        attributions=attributions,
        rejected=rejected,
        timings=sw.timings,
        warnings=warnings,
        stats={
            "n_regions": len(regions),
            "n_confirmed": len(confirmed),
            "n_rejected": len(rejected),
            "segmentation_backend": seg.backend,
            "lookalike_source": lookalike.source,
            "ingest": ingest.stats,
            "segmentation": seg.stats,
            "total_s": round(sum(sw.timings.values()), 3),
        },
    )


def _attribute_one(
    cand: SlickCandidate,
    region: RegionFeatures,
    morph,
    scene: Scene,
    config: Config,
    dcfg: DecisionConfig,
    ais_tracks,
    warnings: list[str],
) -> Attribution:
    """Drift + AIS + scoring for one candidate, with early exits."""
    from attribute.dark_vessel import find_dark_vessels, summarise
    from attribute.scoring import ScoringContext, rank_vessels
    from drift.runner import build_fields, run_backward_drift

    extra: dict[str, Any] = {"morphology_reason": morph.reason}

    # Independent confirmation. A detection coinciding with a documented spill
    # is supported from OUTSIDE the model entirely, which is the strongest
    # confidence signal available - and the main defence against presenting a
    # look-alike as a real spill.
    registry = getattr(config, "_incident_registry", None)
    if registry is not None:
        lon, lat = cand.centroid
        matches = registry.find(lon, lat, scene.acquired_at)
        extra["corroboration"] = {
            "confirmed": bool(matches),
            "n_matches": len(matches),
            "matches": [m.to_dict() for m in matches[:3]],
            "note": (
                "Cross-referenced against documented spill registries. A match "
                "means an independently recorded incident occurred here at "
                "about this time."
            ),
        }
    if morph.matched_source is not None:
        extra["matched_source"] = {
            "name": morph.matched_source.name,
            "kind": morph.matched_source.kind,
            "note": morph.matched_source.note,
        }

    # Grade how sure we are this is actually oil, combining the physics with
    # the independent registry. This is what the UI filters on, so a weakly
    # supported patch is never presented as a spill.
    from detect.confidence import assess

    assessment = assess(
        p_oil=cand.p_oil,
        wind_window_score=cand.wind.window_score,
        damping_ratio=cand.damping_ratio,
        elongation=cand.elongation,
        area_km2=cand.area_km2,
        morphology=cand.morphology,
        rejected_reason=cand.rejected_reason,
        corroboration=extra.get("corroboration"),
        source_type=morph.source_type,
    )
    extra["confidence"] = assessment.to_dict()

    # Rejected look-alikes and fixed sources never reach vessel attribution.
    if cand.is_rejected or morph.source_type in ("natural_seep", "infrastructure"):
        return decide(cand, None, morph.source_type, [], dcfg, extra)

    backtrack_hours = float(config.get("drift.backtrack_hours", 12.0))
    try:
        currents, wind_field = build_fields(
            config,
            wind_speed_ms=cand.wind.speed_ms,
            wind_direction_deg=cand.wind.direction_deg,
        )
        drift = run_backward_drift(
            candidate=cand,
            observed_at=scene.acquired_at,
            currents=currents,
            wind=wind_field,
            backtrack_hours=backtrack_hours,
            timestep_minutes=float(config.get("drift.timestep_minutes", 30.0)),
            n_particles=int(config.get("drift.n_particles", 500)),
            diffusion_m2_s=float(config.get("drift.diffusion_m2_s", 5.0)),
            backend=str(config.get("drift.backend", "auto")),
        )
        origin = drift.origin
        warnings.extend(drift.warnings)
        extra["drift_warnings"] = drift.warnings
        extra["drift_stats"] = drift.stats
    except Exception as exc:
        # Never substitute a plausible origin for a failed run (rule 6).
        log.error("Drift failed for %s: %s", cand.candidate_id, exc)
        warnings.append(f"drift failed for {cand.candidate_id}: {exc}")
        return decide(cand, None, morph.source_type, [], dcfg, extra)

    release_time = origin.estimated_at
    if ais_tracks is None:
        extra["ais"] = "no AIS source configured"
        return decide(cand, origin, morph.source_type, [], dcfg, extra)

    try:
        tracks = (
            ais_tracks(origin, release_time) if callable(ais_tracks) else list(ais_tracks)
        )
    except Exception as exc:
        log.error("AIS lookup failed for %s: %s", cand.candidate_id, exc)
        warnings.append(f"AIS lookup failed for {cand.candidate_id}: {exc}")
        return decide(cand, origin, morph.source_type, [], dcfg, extra)

    ctx = ScoringContext(
        origin=origin,
        slick_axis_deg=region.orientation_deg if cand.morphology == "linear" else None,
        release_time=release_time,
        search_radius_km=float(config.get("attribute.search_radius_km", 50.0)),
        window_hours=float(config.get("attribute.ais_window_before_h", 8.0)),
        weights=config.get("attribute.weights"),
        dark_vessel_bonus=float(config.get("attribute.dark_vessel_bonus", 0.15)),
        gap_min_minutes=float(config.get("attribute.gap_min_minutes", 30.0)),
    )
    vessels = rank_vessels(tracks, ctx, top_n=int(config.get("attribute.top_n", 3)))

    dark = find_dark_vessels(
        tracks, origin, release_time,
        min_gap_minutes=float(config.get("attribute.gap_min_minutes", 30.0)),
    )
    # Full observed passage per ranked vessel: where the track we hold begins
    # and ends, the nearest port to each, and where the course points next.
    from attribute.voyage import build_voyage

    by_mmsi = {t.mmsi: t for t in tracks}
    voyages: dict[str, Any] = {}
    for candidate in vessels:
        track = by_mmsi.get(candidate.mmsi)
        if track is None:
            continue
        voyage = build_voyage(track, declared_destination=track.meta.get("destination"))
        if voyage is not None:
            voyages[candidate.mmsi] = voyage.to_dict()
    extra["voyages"] = voyages

    extra["dark_vessels"] = summarise(dark)
    extra["ais"] = {
        "n_tracks": len(tracks),
        "window_hours": ctx.window_hours,
        "note": "free AIS lags ~72 h; this is near-real-time attribution, not live",
    }

    return decide(cand, origin, morph.source_type, vessels, dcfg, extra)
