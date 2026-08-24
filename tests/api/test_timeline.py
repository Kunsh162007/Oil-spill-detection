"""The timeline endpoint: origin, observed, now - and the active/past split."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    import api.main as main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def candidate_id(client):
    features = client.get("/api/slicks").json()["features"]
    assert features, "no detections to test against"
    return features[0]["properties"]["candidate_id"]


class TestTimeline:
    def test_returns_the_three_states_in_order(self, client, candidate_id):
        data = client.get(f"/api/slicks/{candidate_id}/timeline").json()
        labels = [s["label"] for s in data["states"]]
        assert "observed" in labels
        assert labels.index("origin") < labels.index("observed") if "origin" in labels else True
        assert labels[-1] in ("now", "historical")

    def test_observed_state_is_marked_as_an_observation(self, client, candidate_id):
        """The middle state is the only one that is not a model output."""
        data = client.get(f"/api/slicks/{candidate_id}/timeline").json()
        observed = next(s for s in data["states"] if s["label"] == "observed")
        assert observed["uncertainty_km"] == 0.0
        assert "observation" in observed["description"].lower()

    def test_modelled_states_carry_uncertainty(self, client, candidate_id):
        data = client.get(f"/api/slicks/{candidate_id}/timeline").json()
        for state in data["states"]:
            if state["label"] != "observed":
                assert state["uncertainty_km"] >= 0.0
                assert state["description"]

    def test_states_are_chronological(self, client, candidate_id):
        data = client.get(f"/api/slicks/{candidate_id}/timeline").json()
        times = [s["at"] for s in data["states"]]
        assert times == sorted(times)

    def test_carries_vessel_tracks_for_the_map(self, client, candidate_id):
        """Selecting a moment must keep the ship routes on screen."""
        data = client.get(f"/api/slicks/{candidate_id}/timeline").json()
        assert "vessels" in data

    def test_states_its_caveat(self, client, candidate_id):
        data = client.get(f"/api/slicks/{candidate_id}/timeline").json()
        assert "weathering" in data["caveat"].lower()

    def test_unknown_slick_404s(self, client):
        assert client.get("/api/slicks/NOPE-999/timeline").status_code == 404

    def test_bad_time_is_rejected(self, client, candidate_id):
        response = client.get(f"/api/slicks/{candidate_id}/timeline?at=not-a-date")
        assert response.status_code == 400


class TestActivePastSplit:
    def test_world_index_reports_the_split(self, client):
        meta = client.get("/api/slicks").json()["meta"]
        assert "n_active" in meta and "n_historical" in meta
        assert meta["n_active"] + meta["n_historical"] == meta["n_slicks"]

    def test_every_detection_is_classified(self, client):
        for f in client.get("/api/slicks").json()["features"]:
            assert f["properties"]["activity"] in ("active", "historical")
            assert f["properties"]["age_hours"] >= 0

    def test_old_scenes_are_historical(self, client):
        """Archived imagery must not imply oil is on the water right now."""
        window = client.get("/api/slicks").json()["meta"]["active_window_hours"]
        for f in client.get("/api/slicks").json()["features"]:
            p = f["properties"]
            if p["age_hours"] > window:
                assert p["activity"] == "historical"

    def test_stale_scenes_get_no_present_position(self, client):
        """Refusing to extrapolate is the point; a wrong position is worse."""
        features = client.get("/api/slicks").json()["features"]
        stale = [f for f in features if f["properties"]["age_hours"] > 72]
        if not stale:
            pytest.skip("no stale detections available")
        cid = stale[0]["properties"]["candidate_id"]
        data = client.get(f"/api/slicks/{cid}/timeline").json()
        assert data["states"][-1]["label"] == "historical"
        assert not data["forward_reliable"]
