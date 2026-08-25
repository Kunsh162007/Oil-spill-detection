"""Analysis store - runs the pipeline once, serves results many times.

The API must not re-run a scene on every map pan. Results are computed on
first request (or at startup) and cached in memory, keyed by scene id.
"""

from __future__ import annotations

import json
import logging
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import Config, load_config, resolve_path
from core.contracts import Scene, to_dict
from decision.pipeline import SceneAnalysis, analyse_scene

log = logging.getLogger(__name__)


def live_analysis_allowed() -> bool:
    """Whether this process may run the pipeline on demand.

    A container sized for serving cached results cannot run an analysis: the
    coastline grid alone is 933 MB resident. Without this switch a single
    "re-analyse" click OOM-kills the service and takes the whole map down with
    it, which is a much worse failure than refusing the request. Defaults to
    allowed, so local runs and scripts are unaffected.
    """
    import os

    return os.getenv("ALLOW_LIVE_ANALYSIS", "true").strip().lower() not in (
        "0", "false", "no", "off")


class LiveAnalysisDisabled(RuntimeError):
    """Raised when a scene has no cached analysis and none may be computed."""


# How many scene analyses to hold in memory at once. The world map is served
# from a prebuilt index, so the only thing that loads an analysis is a user
# clicking one slick; keeping a few around covers the follow-up requests for
# its drift and timeline without letting a browse session grow without bound.
MAX_CACHED_ANALYSES = 3


class AnalysisStore:
    """Thread-safe cache of per-scene analyses, bounded to MAX_CACHED_ANALYSES."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.registry = self._load_registry(config)
        # Insertion-ordered, so the oldest entry is the first key.
        self._analyses: dict[str, SceneAnalysis] = {}
        self._manifests: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._errors: dict[str, str] = {}
        self._cache_errors: dict[str, str] = {}

    @staticmethod
    def _load_registry(config: Config):
        """Documented-incident registry, shared by every scene analysis.

        Absence is not fatal - the pipeline still detects - but corroboration
        and the world incident layer both go quiet, so it is logged loudly.
        """
        from detect.incidents import load_registry

        path = resolve_path(config.get("incidents.path", "data/reference/noaa_incidents.csv"))
        try:
            registry = load_registry(path if path.exists() else None)
        except Exception as exc:
            log.error("Incident registry unavailable (%s). Run scripts/fetch_incidents.py", exc)
            return None
        setattr(config, "_incident_registry", registry)
        log.info("Incident registry ready: %d documented spills", len(registry))
        return registry

    # -- discovery ---------------------------------------------------------

    def discover(self, roots: list[str] | None = None) -> list[dict[str, Any]]:
        """Find scene manifests on disk. A manifest is a JSON file with a
        scene_id, a bbox and a vv_path."""
        roots = roots or [
            "data/live",            # freshly fetched imagery, checked first
            "data/demo_internal", "data/demo_finale", "data/dev", "data/dev/scenes",
        ]
        found: list[dict[str, Any]] = []
        for root in roots:
            base = resolve_path(root)
            if not base.is_dir():
                continue
            for path in sorted(base.glob("*.json")):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                if not all(k in data for k in ("scene_id", "bbox", "vv_path")):
                    continue
                data["_manifest_path"] = str(path)
                self._manifests[data["scene_id"]] = data
                found.append(data)
        log.info("Discovered %d scene manifest(s)", len(found))
        return found

    def manifest(self, scene_id: str) -> dict[str, Any] | None:
        return self._manifests.get(scene_id)

    @property
    def manifests(self) -> dict[str, dict[str, Any]]:
        return dict(self._manifests)

    # -- analysis ----------------------------------------------------------

    def scene_from_manifest(self, data: dict[str, Any]) -> Scene:
        return Scene(
            scene_id=data["scene_id"],
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            bbox=tuple(data["bbox"]),
            vv_path=resolve_path(data["vv_path"]),
            vh_path=resolve_path(data["vh_path"]) if data.get("vh_path") else None,
            orbit_direction=data.get("orbit_direction", "DESCENDING"),
        )

    def _load_precomputed(self, scene_id: str) -> SceneAnalysis | None:
        """A build-time analysis for this scene, if one was baked in.

        Running the pipeline needs far more memory than serving its result. On
        a small container the analysis happens once during the image build and
        only the answer ships inside - polygons, scores, tracks, a few hundred
        KB with no arrays. See scripts/precompute.py.
        """
        path = resolve_path("data/precomputed") / f"{scene_id}.pkl"
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                analysis = pickle.load(fh)
        except Exception as exc:
            # A stale or truncated cache must not take the service down; fall
            # through and analyse live, which may still succeed. The reason is
            # kept because "no cached analysis" and "the cache would not load"
            # look identical from outside and need very different fixes - a
            # cache written on Windows, for instance, carries WindowsPath
            # objects that Linux refuses to instantiate.
            log.warning("Precomputed cache unusable for %s (%s); analysing live",
                        scene_id, exc)
            self._cache_errors[scene_id] = f"{type(exc).__name__}: {exc}"
            return None
        log.info("Loaded precomputed analysis for %s", scene_id)
        return analysis

    def get(self, scene_id: str, force: bool = False) -> SceneAnalysis:
        """Analysis for one scene: precomputed if available, else computed."""
        with self._lock:
            if not force and scene_id in self._analyses:
                return self._analyses[scene_id]

        if not force:
            precomputed = self._load_precomputed(scene_id)
            if precomputed is not None:
                with self._lock:
                    self._remember(scene_id, precomputed)
                return precomputed

        data = self._manifests.get(scene_id)
        if data is None:
            raise KeyError(f"Unknown scene: {scene_id}")

        if not live_analysis_allowed():
            reason = self._cache_errors.get(scene_id)
            detail = (f"its cached analysis failed to load ({reason})"
                      if reason else "it has no cached analysis")
            raise LiveAnalysisDisabled(
                f"Cannot serve {scene_id}: {detail}, and live analysis is "
                f"disabled in this deployment (ALLOW_LIVE_ANALYSIS=false). "
                f"Run scripts/precompute.py and rebuild the image."
            )

        scene = self.scene_from_manifest(data)
        config = self._config_for(data)

        from detect.wind import build_wind_lookup

        wind_lookup = build_wind_lookup(config)
        ais_tracks = self._ais_for(data)

        log.info("Analysing scene %s", scene_id)
        analysis = analyse_scene(
            scene, config, wind_lookup=wind_lookup, ais_tracks=ais_tracks
        )
        with self._lock:
            self._remember(scene_id, analysis)
        return analysis

    def _remember(self, scene_id: str, analysis: SceneAnalysis) -> None:
        """Cache one analysis, evicting the oldest once the bound is reached.

        Caller must hold the lock.
        """
        self._analyses[scene_id] = analysis
        while len(self._analyses) > MAX_CACHED_ANALYSES:
            oldest = next(iter(self._analyses))
            del self._analyses[oldest]

    def _config_for(self, data: dict[str, Any]) -> Config:
        """Per-scene config override, when the manifest names one."""
        cfg_name = data.get("config")
        if not cfg_name:
            return self.config
        config = load_config(cfg_name)
        # The override is a fresh Config object, so it needs the registry too;
        # without this, corroboration silently vanishes for those scenes.
        if self.registry is not None:
            setattr(config, "_incident_registry", self.registry)
        return config

    def _ais_for(self, data: dict[str, Any]):
        """AIS source for a scene: a local CSV, GFW, or none."""
        ais_path = data.get("ais_path")
        if ais_path:
            from attribute.ais import load_ais_csv

            path = resolve_path(ais_path)
            try:
                tracks = load_ais_csv(path, source=str(path.name))
            except (FileNotFoundError, ValueError) as exc:
                log.error("AIS load failed for %s: %s", data["scene_id"], exc)
                return None
            return lambda origin, when: tracks
        return None

    def analysed_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._analyses)
