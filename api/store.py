"""Analysis store - runs the pipeline once, serves results many times.

The API must not re-run a scene on every map pan. Results are computed on
first request (or at startup) and cached in memory, keyed by scene id.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import Config, load_config, resolve_path
from core.contracts import Scene, to_dict
from decision.pipeline import SceneAnalysis, analyse_scene

log = logging.getLogger(__name__)


class AnalysisStore:
    """Thread-safe cache of per-scene analyses."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._analyses: dict[str, SceneAnalysis] = {}
        self._manifests: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._errors: dict[str, str] = {}

    # -- discovery ---------------------------------------------------------

    def discover(self, roots: list[str] | None = None) -> list[dict[str, Any]]:
        """Find scene manifests on disk. A manifest is a JSON file with a
        scene_id, a bbox and a vv_path."""
        roots = roots or ["data/demo_internal", "data/demo_finale", "data/dev"]
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

    def get(self, scene_id: str, force: bool = False) -> SceneAnalysis:
        """Analysis for one scene, computed on first use."""
        with self._lock:
            if not force and scene_id in self._analyses:
                return self._analyses[scene_id]

        data = self._manifests.get(scene_id)
        if data is None:
            raise KeyError(f"Unknown scene: {scene_id}")

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
            self._analyses[scene_id] = analysis
        return analysis

    def _config_for(self, data: dict[str, Any]) -> Config:
        """Per-scene config override, when the manifest names one."""
        cfg_name = data.get("config")
        return load_config(cfg_name) if cfg_name else self.config

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
