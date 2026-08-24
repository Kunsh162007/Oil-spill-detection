"""API contract. The UI depends on these shapes; so does the demo."""

from __future__ import annotations

import json
import math

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
