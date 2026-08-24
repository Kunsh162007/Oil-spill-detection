"""YAML config loading, with env-var interpolation and stub selection.

Every module reads its own section. The `stubs:` block decides which stages
run for real and which return valid fake output — that switch is what keeps
a demo alive when one stage dies. See CLAUDE.md, "STUBS ARE PERMANENT".
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

DEFAULTS: dict[str, Any] = {
    "stubs": {
        # False => run the real implementation.
        "ingest": False,
        "stage_a": False,
        "stage_b": True,      # no trained checkpoint yet; see scripts/train.py
        "lookalike": False,
        "morphology": False,
        "drift": False,
        "ais": False,
        "wind": False,
    },
    "ingest": {
        "target_resolution_m": 80.0,   # 8x downsample from S1 GRD IW ~10 m
        "tile_size": 512,
        "tile_overlap": 0.25,
        "speckle_filter": "refined_lee",
        "speckle_window": 7,
        "cache_dir": "data/cache",
    },
    "detect": {
        "stage_a_threshold": 0.15,
        "stage_b_threshold": 0.5,
        "min_area_km2": 0.05,
        "checkpoint": "models/stage_b.pt",
        "encoder": "resnet34",
        "arch": "unet",
    },
    "lookalike": {
        "p_oil_threshold": 0.5,
        "model_path": "models/lookalike.joblib",
    },
    "drift": {
        "backend": "auto",          # auto | opendrift | analytical
        "backtrack_hours": 12.0,
        "timestep_minutes": 30.0,
        "n_particles": 500,
        "diffusion_m2_s": 5.0,
        "wind_drift_factor": 0.03,  # 3% of wind speed, standard for surface oil
    },
    "attribute": {
        "ais_window_before_h": 8.0,
        "ais_window_after_h": 6.0,
        "search_radius_km": 50.0,
        "top_n": 3,
        "weights": {"parity": 0.3, "proximity": 0.4, "temporality": 0.3},
        "dark_vessel_bonus": 0.15,
        "gap_min_minutes": 30.0,
    },
    "decision": {
        "abstain_margin": 0.08,      # top-two within this => insufficient evidence
        "min_top_score": 0.35,
        "min_wind_window_score": 0.15,
    },
    "api": {"host": "127.0.0.1", "port": 8000},
}


def _interpolate(value: Any) -> Any:
    """Expand ${VAR} and ${VAR:-default} from the environment."""
    if isinstance(value, str):
        def sub(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) or "")
        return _ENV_PATTERN.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge; override wins. Returns a new dict, mutates neither."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Dotted-path access over the merged config tree."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, path: str) -> Any:
        """Like get(), but raises rather than silently returning None.

        Used for anything whose absence would make the stage produce
        plausible-looking nonsense.
        """
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise KeyError(f"Required config key missing: {path!r}")
        return value

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.get(name, {}) or {})

    def use_stub(self, module: str) -> bool:
        return bool(self.get(f"stubs.{module}", False))

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def __repr__(self) -> str:
        return f"Config(sections={sorted(self._data)})"


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config layered over DEFAULTS.

    A missing path is an error, not a silent fall-back to defaults — a typo'd
    config filename that quietly runs the default pipeline is exactly the kind
    of thing that wastes an afternoon.
    """
    data = DEFAULTS
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping, got {type(loaded).__name__}: {p}")
        data = deep_merge(DEFAULTS, _interpolate(loaded))
    return Config(data)


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p
