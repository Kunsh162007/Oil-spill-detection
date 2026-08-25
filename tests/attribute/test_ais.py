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
