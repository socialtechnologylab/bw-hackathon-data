"""Tests for GFS area-mean aggregation and U/V → wind magnitude.

Herbie's network behaviour is mocked; the unit tests cover only the
deterministic transformations: spatial subset, area-mean, U/V combine.
"""

from datetime import UTC, datetime

import numpy as np
import xarray as xr

from bw_hackathon_data.gfs import (
    aggregate_to_belgium_means,
    combine_uv_to_speed,
)


def _grid(values_2d: np.ndarray, lats: np.ndarray, lons: np.ndarray, name: str) -> xr.DataArray:
    """Build a 2D DataArray on a (lat, lon) grid."""
    return xr.DataArray(
        values_2d,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
        name=name,
    )


def test_combine_uv_to_speed_magnitude():
    """sqrt(u^2 + v^2). u=3, v=4 → 5."""
    u = np.array([[3.0, 0.0], [1.0, -3.0]])
    v = np.array([[4.0, 0.0], [0.0, 4.0]])
    result = combine_uv_to_speed(u, v)
    np.testing.assert_allclose(result, [[5.0, 0.0], [1.0, 5.0]])


def test_aggregate_to_belgium_means_simple_grid():
    """One forecast hour, single variable, hand-built grid → area-mean."""
    lats = np.array([49.0, 50.0, 51.0, 52.0])
    lons = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    # Constant field = 100 everywhere → mean = 100 regardless of bbox.
    field = np.full((len(lats), len(lons)), 100.0)
    ds = xr.Dataset({"dswrf": _grid(field, lats, lons, "dswrf")})
    ds = ds.expand_dims({"step": [np.timedelta64(0, "h")]})

    df = aggregate_to_belgium_means(
        ds,
        cycle=datetime(2025, 1, 1, 0, tzinfo=UTC),
        bbox_lat=(49.5, 51.5),
        bbox_lon=(2.5, 6.5),
        var_rename={"dswrf": "ghi_fcst"},
    )

    assert df.columns == ["cycle_utc", "valid_time", "fxx", "ghi_fcst"]
    assert df.height == 1
    assert df["ghi_fcst"][0] == 100.0
    assert df["fxx"][0] == 0


def test_aggregate_to_belgium_means_picks_only_inside_bbox():
    """Points outside the bbox must NOT contribute."""
    lats = np.array([49.0, 50.0, 51.0, 52.0])  # 49 & 52 outside; 50 & 51 inside
    lons = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])  # 2 & 7 outside; 3–6 inside
    # Field is 1.0 outside the bbox, 99.0 inside. If aggregation works,
    # the result should be exactly 99.0.
    field = np.ones((len(lats), len(lons)))
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if 49.5 <= lat <= 51.5 and 2.5 <= lon <= 6.5:
                field[i, j] = 99.0
    ds = xr.Dataset({"dswrf": _grid(field, lats, lons, "dswrf")})
    ds = ds.expand_dims({"step": [np.timedelta64(0, "h")]})

    df = aggregate_to_belgium_means(
        ds,
        cycle=datetime(2025, 1, 1, 0, tzinfo=UTC),
        bbox_lat=(49.5, 51.5),
        bbox_lon=(2.5, 6.5),
        var_rename={"dswrf": "ghi_fcst"},
    )
    assert df["ghi_fcst"][0] == 99.0


def test_aggregate_to_belgium_means_multi_step():
    """Two forecast hours → two output rows ordered by fxx."""
    lats = np.array([50.0, 51.0])
    lons = np.array([3.0, 4.0])
    field_hr0 = np.full((2, 2), 100.0)
    field_hr6 = np.full((2, 2), 200.0)
    arr = np.stack([field_hr0, field_hr6])  # shape (step, lat, lon)
    ds = xr.Dataset(
        {"dswrf": (("step", "latitude", "longitude"), arr)},
        coords={
            "step": [np.timedelta64(0, "h"), np.timedelta64(6, "h")],
            "latitude": lats,
            "longitude": lons,
        },
    )

    df = aggregate_to_belgium_means(
        ds,
        cycle=datetime(2025, 1, 1, 0, tzinfo=UTC),
        bbox_lat=(49.5, 51.5),
        bbox_lon=(2.5, 6.5),
        var_rename={"dswrf": "ghi_fcst"},
    )
    assert df.height == 2
    assert df["fxx"].to_list() == [0, 6]
    assert df["ghi_fcst"].to_list() == [100.0, 200.0]
    assert df["valid_time"][0] == "2025-01-01T00:00:00+00:00"
    assert df["valid_time"][1] == "2025-01-01T06:00:00+00:00"
