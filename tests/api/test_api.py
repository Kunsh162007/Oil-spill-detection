"""API contract. The UI depends on these shapes; so does the demo."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client(request):
    import api.main as main

    with TestClient(main.app) as c:
        yield c


class TestHealth:
    def test_reports_which_backends_are_real(self, client):
        """A stubbed stage must never be reported as a trained model."""
        data = client.get("/api/health").json()
        assert data["status"] == "ok"
        assert "segmentation" in data["backends"]
        assert "drift" in data["backends"]

    def test_states_the_timeliness_limit(self, client):
        """CLAUDE.md rule 2: never say real-time."""
        data = client.get("/api/health").json()
        assert "near-real-time" in data["timeliness"]
        blob = json.dumps(data).lower()
        assert "real-time" not in blob.replace("near-real-time", "")


class TestWorldIndex:
    def test_lists_confirmed_slicks(self, client):
        data = client.get("/api/slicks").json()
        assert data["type"] == "FeatureCollection"
        assert data["meta"]["n_slicks"] >= 1

    def test_rejected_lookalikes_stay_off_the_world_map(self, client):
        data = client.get("/api/slicks").json()
        assert all(f["properties"]["is_oil"] for f in data["features"])

    def test_carries_the_disclaimer(self, client):
        assert "not evidence" in client.get("/api/slicks").json()["meta"]["disclaimer"]

    def test_geojson_is_lon_lat_ordered(self, client):
        """Lat/lon inversion silently lands everything in the wrong ocean.

        Checked against each detection's own scene bbox rather than a fixed
        region, so the test stays valid as scenes are added anywhere on Earth.
        A swapped pair falls outside its scene almost every time.
        """
        scenes = {
            s["scene_id"]: s["bbox"]
            for s in client.get("/api/scenes").json()["scenes"]
            if s.get("bbox")
        }
        checked = 0
        for feature in client.get("/api/slicks").json()["features"]:
            geometry = feature["geometry"]
            coords = geometry["coordinates"]
            lon, lat = coords[0][0] if geometry["type"] == "Polygon" else coords
            assert -180 <= lon <= 180, f"longitude {lon} out of range"
            assert -90 <= lat <= 90, f"latitude {lat} out of range"

            bbox = scenes.get(feature["properties"]["scene_id"])
            if bbox:
                pad = 0.5  # slicks may sit slightly outside a nominal bbox
                assert bbox[0] - pad <= lon <= bbox[2] + pad, (
                    f"lon {lon} outside scene bbox {bbox} - coordinates may be swapped"
                )
                assert bbox[1] - pad <= lat <= bbox[3] + pad, (
                    f"lat {lat} outside scene bbox {bbox} - coordinates may be swapped"
                )
                checked += 1
        assert checked, "no detection could be checked against a scene bbox"

    def test_payload_is_strict_json(self, client):
        """NaN is not valid JSON and breaks the browser parser."""
        raw = client.get("/api/slicks").text
        json.loads(raw)  # strict by default: NaN would raise
        assert "NaN" not in raw and "Infinity" not in raw


class TestSceneEndpoint:
    def test_scene_includes_rejected_lookalikes_and_reasons(self, client):
        """Whichever scene has rejections must explain every one of them.

        Not every scene contains look-alikes, so this searches for one that
        does rather than assuming a particular scene sorts first.
        """
        checked = False
        for scene in client.get("/api/scenes").json()["scenes"]:
            data = client.get(f"/api/scenes/{scene['scene_id']}").json()
            rejected = [f for f in data["features"] if not f["properties"]["is_oil"]]
            if not rejected:
                continue
            assert all(f["properties"]["rejected_reason"] for f in rejected)
            checked = True
            break
        assert checked, "no scene contained a rejected look-alike to inspect"

    def test_unknown_scene_404s(self, client):
        assert client.get("/api/scenes/NO_SUCH_SCENE").status_code == 404


class TestDetailAndBacktrace:
    @pytest.fixture(scope="class")
    def candidate_id(self, client):
        """Any detection - used for tests that do not need AIS."""
        return client.get("/api/slicks").json()["features"][0]["properties"]["candidate_id"]

    @pytest.fixture(scope="class")
    def attributed_id(self, client):
        """A detection that actually has ranked vessels behind it.

        Scenes without AIS coverage correctly abstain and rank nobody, so a
        vessel test has to seek out one that had tracks to work with.
        """
        for feature in client.get("/api/slicks").json()["features"]:
            cid = feature["properties"]["candidate_id"]
            detail = client.get(f"/api/slicks/{cid}").json()
            if detail.get("vessels"):
                return cid
        pytest.skip("no detection has AIS coverage; run scripts/fetch_ais.py")

    def test_detail_ranks_vessels(self, client, attributed_id):
        data = client.get(f"/api/slicks/{attributed_id}").json()
        assert data["vessels"]
        assert [v["rank"] for v in data["vessels"]] == list(range(1, len(data["vessels"]) + 1))

    def test_each_vessel_has_readable_evidence(self, client, attributed_id):
        for vessel in client.get(f"/api/slicks/{attributed_id}").json()["vessels"]:
            assert len(vessel["evidence"]) > 30
            for key in ("parity", "proximity", "temporality"):
                assert 0.0 <= vessel[key] <= 1.0

    def test_backtrace_frames_run_oldest_first(self, client, candidate_id):
        """The UI plays them forward, so the oil must travel the way it did."""
        data = client.get(f"/api/slicks/{candidate_id}/backtrace").json()
        frames = data["frames"]
        assert len(frames) > 2
        assert frames[0]["hours_before"] > frames[-1]["hours_before"]
        assert frames[-1]["hours_before"] == 0.0

    def test_every_frame_carries_a_timestamp(self, client, candidate_id):
        """The demo shows the date and time as the oil crawls back."""
        for frame in client.get(f"/api/slicks/{candidate_id}/backtrace").json()["frames"]:
            assert "T" in frame["at"]
            assert -180 <= frame["lon"] <= 180

    def test_backtrace_states_its_limits(self, client, candidate_id):
        data = client.get(f"/api/slicks/{candidate_id}/backtrace").json()
        assert "weathering" in data["caveat"].lower()
        assert data["uncertainty_km"] > 0

    def test_vessel_tracks_included_for_the_map(self, client, attributed_id):
        data = client.get(f"/api/slicks/{attributed_id}/backtrace").json()
        assert any(len(v["track"]) > 1 for v in data["vessels"])

    def test_unknown_slick_404s(self, client):
        assert client.get("/api/slicks/NOPE-999").status_code == 404


class TestUI:
    def test_index_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Oil Spill Detection" in response.text

    def test_static_assets_are_served(self, client):
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/app.css").status_code == 200


# -- cached world index ------------------------------------------------------
#
# The deployed container serves the map from a prebuilt index rather than
# deserialising every scene's analysis per request; holding all of them at once
# is what exhausted a 512 MB worker. These cover the part of that cache which
# can go wrong silently: a frozen "active" flag would tell a viewer that a
# months-old detection is current oil.

def test_refresh_ages_recomputes_activity_from_acquisition_time():
    from api.serialize import refresh_ages

    payload = {
        "features": [
            {"properties": {"acquired_at": "2020-01-01T00:00:00+00:00",
                            "activity": "active", "age_hours": 1.0}},
        ],
        "meta": {"n_active": 1, "n_historical": 0},
    }

    refreshed = refresh_ages(payload)

    props = refreshed["features"][0]["properties"]
    assert props["activity"] == "historical"
    assert props["age_hours"] > 24_000
    assert refreshed["meta"]["n_active"] == 0
    assert refreshed["meta"]["n_historical"] == 1


def test_refresh_ages_keeps_a_recent_detection_active():
    from datetime import datetime, timedelta, timezone

    from api.serialize import refresh_ages

    recent = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
    payload = {"features": [{"properties": {"acquired_at": recent}}], "meta": {}}

    refreshed = refresh_ages(payload)

    assert refreshed["features"][0]["properties"]["activity"] == "active"
    assert refreshed["meta"]["n_active"] == 1


def test_refresh_ages_tolerates_a_feature_with_no_acquisition_time():
    """A feature without a timestamp must not take the whole map down."""
    from api.serialize import refresh_ages

    payload = {"features": [{"properties": {"scene_id": "X"}}], "meta": {}}

    refreshed = refresh_ages(payload)

    assert refreshed["meta"]["n_active"] == 0


def test_store_evicts_analyses_beyond_the_cache_bound():
    from api.store import MAX_CACHED_ANALYSES, AnalysisStore

    store = AnalysisStore.__new__(AnalysisStore)
    store._analyses = {}

    for i in range(MAX_CACHED_ANALYSES + 4):
        store._remember(f"scene-{i}", object())

    assert len(store._analyses) == MAX_CACHED_ANALYSES
    # The most recent survive; the oldest are gone.
    assert "scene-0" not in store._analyses
    assert f"scene-{MAX_CACHED_ANALYSES + 3}" in store._analyses


# -- memory landmines --------------------------------------------------------

def test_importing_landmask_does_not_load_the_coastline_grid():
    """global_land_mask allocates 933 MB on import.

    That is a 21600 x 43200 boolean array. Importing it at module scope kills
    any container smaller than a gigabyte before it serves a single request, so
    ingest.landmask must defer it to first use. This test fails the moment
    somebody moves the import back to the top of the file.
    """
    import subprocess
    import sys

    code = (
        "import sys; import ingest.landmask; "
        "print('global_land_mask' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parents[2]))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "ingest.landmask imported global_land_mask at module scope; "
        "that is 933 MB resident before any work is done"
    )


def test_live_analysis_can_be_switched_off(monkeypatch):
    from api.store import live_analysis_allowed

    monkeypatch.delenv("ALLOW_LIVE_ANALYSIS", raising=False)
    assert live_analysis_allowed() is True

    for value in ("false", "FALSE", "0", "no", "off"):
        monkeypatch.setenv("ALLOW_LIVE_ANALYSIS", value)
        assert live_analysis_allowed() is False, value

    monkeypatch.setenv("ALLOW_LIVE_ANALYSIS", "true")
    assert live_analysis_allowed() is True


def test_health_does_not_import_torch(client):
    """A health check must stay cheap; importing torch costs ~160 MB."""
    import sys

    sys.modules.pop("torch", None)
    body = client.get("/api/health").json()

    assert "torch" not in sys.modules
    assert "memory" in body
    assert "torch" not in body["memory"]["heavy_imports"]


def test_precomputed_analyses_use_portable_paths():
    """A cache written on Windows must still load on Linux.

    pathlib pickles the concrete class, so a Path built on Windows arrives as
    pathlib.WindowsPath and a Linux container refuses to instantiate it. That
    fails at load time for every entry, and the service then looks as though it
    simply has no cache - a very different problem from the real one.
    """
    import pathlib
    import pickle

    cache_dir = Path(__file__).resolve().parents[2] / "data" / "precomputed"
    pickles = sorted(cache_dir.glob("*.pkl"))
    if not pickles:
        pytest.skip("no precomputed cache in this checkout")

    for path in pickles:
        with path.open("rb") as fh:
            analysis = pickle.load(fh)
        vv = analysis.scene.vv_path
        assert not isinstance(vv, (pathlib.WindowsPath, pathlib.PosixPath)), (
            f"{path.name} stores a platform-bound {type(vv).__name__}; "
            f"run scripts/precompute.py to normalise it"
        )
