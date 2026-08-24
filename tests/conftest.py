"""Shared fixtures.

Tests use real files from data/, never in-memory fakes for the IO paths -
CLAUDE.md coding standards. The synthetic scene is generated on demand so a
fresh clone can run the suite without downloading anything.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEMO_DIR = REPO_ROOT / "data" / "demo_internal"
DEMO_NAME = "SYNTH_DEMO_001"


@pytest.fixture(scope="session")
def demo_manifest() -> dict:
    """The synthetic demo scene, generated if it does not already exist."""
    manifest_path = DEMO_DIR / f"{DEMO_NAME}.json"
    if not manifest_path.exists():
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "make_demo_scene.py")],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def demo_scene(demo_manifest):
    from core.contracts import Scene

    return Scene(
        scene_id=demo_manifest["scene_id"],
        acquired_at=datetime.fromisoformat(demo_manifest["acquired_at"]),
        bbox=tuple(demo_manifest["bbox"]),
        vv_path=REPO_ROOT / demo_manifest["vv_path"],
        vh_path=REPO_ROOT / demo_manifest["vh_path"],
        orbit_direction=demo_manifest["orbit_direction"],
    )


@pytest.fixture(scope="session")
def demo_config():
    from core.config import load_config

    return load_config("configs/demo_synthetic.yaml")


@pytest.fixture(scope="session")
def demo_analysis(demo_scene, demo_config, demo_manifest):
    """One full pipeline run, shared across tests - it is the slow fixture."""
    from attribute.ais import load_ais_csv
    from decision.pipeline import analyse_scene
    from detect.wind import build_wind_lookup

    tracks = load_ais_csv(REPO_ROOT / demo_manifest["ais_path"], source="test")
    return analyse_scene(
        demo_scene,
        demo_config,
        wind_lookup=build_wind_lookup(demo_config),
        ais_tracks=lambda origin, when: tracks,
    )
