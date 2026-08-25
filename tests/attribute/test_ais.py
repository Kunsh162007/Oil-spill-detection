"""What the AIS sources can actually supply.

Separate from the scoring tests: these are about the shape of the data that
reaches the scorer, and the ways a source can appear to work while supplying
nothing usable.
"""

from __future__ import annotations

import pytest


class TestGFWLimits:
    """What the public GFW API can and cannot supply.

    Recorded as tests because the gap is easy to miss: /vessels/search answers
    happily, so the integration looks healthy right up to the point where every
    candidate is disqualified for having no positions.
    """

    def test_a_track_without_positions_scores_zero_and_is_disqualified(self):
        """The safe half: no positions means abstain, never a false lead."""
        from datetime import datetime, timezone

        from attribute.ais import VesselTrack
        from attribute.scoring import ScoringContext, disqualify, score_vessel
        from core.contracts import DriftOrigin

        origin = DriftOrigin(lon=75.5, lat=9.4,
                             estimated_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
                             uncertainty_km=20.0, n_particles=100)
        ctx = ScoringContext(origin=origin, slick_axis_deg=90.0,
                             release_time=origin.estimated_at)

        candidate = score_vessel(
            VesselTrack(mmsi="123456789", positions=[], name="NO FIXES",
                        source="gfw-identity"),
            ctx,
        )

        assert candidate.score == 0.0
        assert candidate.parity == candidate.proximity == candidate.temporality == 0.0
        assert disqualify(candidate) is not None

    def test_identity_search_refuses_rather_than_returning_useless_tracks(self):
        """The unsafe half, closed: it must not look like a working query."""
        import attribute.ais as ais_module

        client = ais_module.GFWClient.__new__(ais_module.GFWClient)
        client.token, client.timeout = "test-token", 5.0

        class FakeResponse:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"entries": [{"ssvid": "111111111", "shipname": "A"}]}

        import sys
        import types
        fake_requests = types.ModuleType("requests")
        fake_requests.get = lambda *a, **k: FakeResponse()
        saved = sys.modules.get("requests")
        sys.modules["requests"] = fake_requests
        try:
            from datetime import datetime, timezone
            with pytest.raises(NotImplementedError, match="no track positions"):
                client.fetch_tracks(
                    (75.0, 9.0, 76.0, 10.0),
                    datetime(2026, 8, 24, tzinfo=timezone.utc),
                    datetime(2026, 8, 25, tzinfo=timezone.utc),
                )
        finally:
            if saved is not None:
                sys.modules["requests"] = saved
            else:
                del sys.modules["requests"]

    def test_gap_events_map_onto_our_gap_record(self):
        """The one thing GFW gives that no free positional feed does.

        A gap event carries a single position - where the transponder went
        quiet - plus a bounding box spanning the silence, so the resume point is
        the far corner of that box. Shape verified against a real response.
        """
        import sys
        import types
        from datetime import datetime, timezone

        import attribute.ais as ais_module

        client = ais_module.GFWClient.__new__(ais_module.GFWClient)
        client.token, client.timeout = "test-token", 5.0

        entry = {
            "start": "2026-08-16T01:31:04.000Z",
            "end": "2026-08-18T19:58:36.000Z",
            "type": "gap",
            "position": {"lat": 8.36, "lon": 80.93},
            "boundingBox": [80.5, 8.0, 81.9, 9.4],
            "vessel": {"ssvid": "636026512", "name": "DARK RUNNER"},
            "gap": {"intentionalDisabling": True},
        }
        outside = dict(entry, position={"lat": -40.0, "lon": 10.0})

        class FakeResponse:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"entries": [entry, outside], "nextOffset": None}

        fake_requests = types.ModuleType("requests")
        fake_requests.get = lambda *a, **k: FakeResponse()
        saved = sys.modules.get("requests")
        sys.modules["requests"] = fake_requests
        try:
            gaps = client.fetch_gap_events(
                (65.0, 5.0, 90.0, 25.0),
                datetime(2026, 8, 15, tzinfo=timezone.utc),
                datetime(2026, 8, 25, tzinfo=timezone.utc),
            )
        finally:
            if saved is not None:
                sys.modules["requests"] = saved
            else:
                del sys.modules["requests"]

        assert len(gaps) == 1, "the out-of-bbox event must be filtered out"
        gap = gaps[0]
        assert gap.mmsi == "636026512"
        assert gap.name == "DARK RUNNER"
        assert (gap.last_lon, gap.last_lat) == (80.93, 8.36)
        # Far corner of the silence envelope, not the near one.
        assert (gap.resume_lon, gap.resume_lat) == (81.9, 9.4)
        assert gap.duration_hours > 60
