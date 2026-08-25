"""Current and wind field readers: what they return, and when they refuse."""

from __future__ import annotations

import pytest

xr = pytest.importorskip("xarray")


class TestNetCDFFieldInterpolation:
    """The field is preloaded and interpolated by hand rather than via xarray.

    xarray.interp realigns dimensions and builds an intermediate Dataset on
    every call, and a drift run samples once per particle per timestep. That
    put one real-currents scene at 620 s against a two-minute target; the hand
    interpolation brought the same scene to 4.6 s with identical output. These
    tests pin the behaviour that made the swap safe.
    """

    @staticmethod
    def _field(tmp_path, lats=(0.0, 1.0), lons=(10.0, 11.0)):
        import numpy as np
        import xarray as xr

        from drift.readers import NetCDFField

        times = np.array(["2026-01-01T00:00", "2026-01-01T01:00"], dtype="datetime64[ns]")
        # u varies only with longitude, v only with latitude, so a bilinear
        # result is predictable by hand.
        u = np.zeros((2, len(lats), len(lons)))
        v = np.zeros((2, len(lats), len(lons)))
        for j, _ in enumerate(lats):
            for i, _ in enumerate(lons):
                u[:, j, i] = float(i)
                v[:, j, i] = float(j)
        ds = xr.Dataset(
            {"uo": (("time", "latitude", "longitude"), u),
             "vo": (("time", "latitude", "longitude"), v)},
            coords={"time": times, "latitude": list(lats), "longitude": list(lons)},
        )
        path = tmp_path / "field.nc"
        ds.to_netcdf(path)
        return NetCDFField(path)

    def test_midpoint_is_the_average_of_its_corners(self, tmp_path):
        from datetime import datetime, timezone

        field = self._field(tmp_path)

        sample = field.sample(10.5, 0.5, datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc))

        assert sample.u == pytest.approx(0.5)
        assert sample.v == pytest.approx(0.5)

    def test_corners_are_exact(self, tmp_path):
        from datetime import datetime, timezone

        field = self._field(tmp_path)
        when = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        assert field.sample(10.0, 0.0, when).u == pytest.approx(0.0)
        assert field.sample(11.0, 0.0, when).u == pytest.approx(1.0)
        assert field.sample(10.0, 1.0, when).v == pytest.approx(1.0)

    def test_outside_the_domain_raises_rather_than_clamping(self, tmp_path):
        """A run silently pinned to the boundary gives a confident wrong origin."""
        from datetime import datetime, timezone

        field = self._field(tmp_path)
        when = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="outside"):
            field.sample(20.0, 0.5, when)
        with pytest.raises(ValueError, match="outside"):
            field.sample(10.5, 9.0, when)

    def test_a_time_outside_the_window_raises(self, tmp_path):
        from datetime import datetime, timezone

        field = self._field(tmp_path)

        with pytest.raises(ValueError, match="outside"):
            field.sample(10.5, 0.5, datetime(2030, 1, 1, tzinfo=timezone.utc))

    def test_descending_latitude_is_handled(self, tmp_path):
        """ERA5 and friends publish latitude north-to-south."""
        from datetime import datetime, timezone

        field = self._field(tmp_path, lats=(1.0, 0.0))

        sample = field.sample(10.5, 0.5, datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))

        assert sample.u == pytest.approx(0.5)
        assert sample.v == pytest.approx(0.5)
